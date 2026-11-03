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
  first->linearization_controls[0] = {0.11f, 0.22f};
  first->unscaled_proposal_controls[0] = {0.31f, 0.42f};
  first->proposal_controls[0] = {0.21f, 0.32f};
  first->source_rollout_controls[0] = {0.19f, 0.29f};
  first->proposal_controls_valid = 1;
  first->source_rollout_controls_valid = 1;
  second->linearization_controls[0] = {-0.11f, 0.12f};
  second->unscaled_proposal_controls[0] = {-0.31f, 0.52f};
  second->proposal_controls[0] = {-0.21f, 0.42f};
  second->source_rollout_controls[0] = {-0.19f, 0.39f};
  second->proposal_controls_valid = 1;
  second->source_rollout_controls_valid = 1;
  MpcRtiState_t initial{};
  initial.plant.e_y = -0.17f;
  initial.plant.u = 4.25f;
  initial.plant.delayed_steering_command_1 = -0.06f;
  initial.plant.delayed_steering_command_2 = 0.08f;
  initial.plant.delayed_throttle_command_1 = 0.17f;
  initial.plant.delayed_throttle_command_2 = 0.21f;
  initial.plant.source_plant_initialized = 1;
  initial.plant.source_suspension_history_initialized = 1;
  initial.plant.source_imu_state_initialized = 1;
  initial.plant.source_plant_internal_state[0] = 71.0;
  initial.incoming_branch_flags = 5;
  initial.incoming_wheel_slip_excess_mps[3] = 0.031;
  MpcRtiCycleResult_t result{};
  result.nominal_control_count = RICCATI_HORIZON;
  result.nominal_controls[0].steering_target_rad = 0.3f;
  result.nominal_controls[0].throttle_command = 0.25f;
  result.admm_status = 1;
  result.primal_residual = 0.5f;
  result.comparison_objective.valid = 1;
  result.comparison_objective.horizon = PREDICTION_HORIZON;
  result.comparison_objective.successor_recourse_valid=1;
  result.comparison_objective.successor_recourse_horizon=8;
  result.comparison_objective.successor_recourse_weight=4000.f;
  result.comparison_objective.successor_recourse_reserve_m=.04f;
  result.comparison_objective.successor_recourse_dt=.025f;
  result.comparison_objective.successor_recourse_deltas[3]={.02f,-.07f};
  result.r1_step_metrics.valid = 1;
  result.r1_step_metrics.qp_predicted_cost_reduction = 101.0;
  result.r1_step_metrics.state_error_by_stage[12] = 0.045f;
  MpcRtiCycleConfiguration_t cycle_configuration{};
  cycle_configuration.terminal_viability = 1;
  cycle_configuration.model.active_speed_ceiling_mps = 5.0f;
  cycle_configuration.model.weight_source_corridor_reserve = 12345.f;
  cycle_configuration.model.weight_source_lateral_slip = 500.f;
  MpcTrajectorySample_t trajectory[3]{};
  trajectory[0].s = 0.0;
  trajectory[1].s = 1.0;
  trajectory[2].s = 2.0;
  trajectory[1].left_bound = trajectory[1].right_bound = 1.0;
  {
    f1tenth_mpc::MpcQpFailureRecorder recorder(base);
    recorder.submit(1793573160746756676LL, 59.2778, 16.0f,
        initial, MPC_RTI_CYCLE_REJECTED_NONLINEAR_ROLLOUT,
        result, cycle_configuration, trajectory, 3, 2.0, 0.025f,
        first.get(), second.get());
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
  EXPECT_NE(line.find("plant_initialized"), std::string::npos);
  ASSERT_TRUE(std::getline(state, line));
  EXPECT_EQ(std::count(line.begin(), line.end(), ','), 61);
  std::ifstream metadata(entry / "context.json");
  ASSERT_TRUE(std::getline(metadata, line));
  EXPECT_NE(line.find("\"r2_captured\":true"), std::string::npos);
  sdu_apex::qp_fixture::FailureReplayContext source{};
  ASSERT_TRUE(sdu_apex::qp_fixture::read_failure_context(
      (entry / "source_context.mctx").string(), &source));
  EXPECT_FLOAT_EQ(source.initial_state.plant.delayed_steering_command_1,
      -0.06f);
  EXPECT_FLOAT_EQ(source.initial_state.plant.delayed_throttle_command_2,
      0.21f);
  EXPECT_EQ(source.initial_state.plant.source_suspension_history_initialized,
      1);
  EXPECT_EQ(source.initial_state.incoming_branch_flags, 5u);
  EXPECT_DOUBLE_EQ(source.initial_state.incoming_wheel_slip_excess_mps[3],
      0.031);
  EXPECT_FLOAT_EQ(source.r1_proposal[0].steering_target_rad, 0.21f);
  EXPECT_FLOAT_EQ(source.r2_proposal[0].steering_target_rad, -0.21f);
  EXPECT_DOUBLE_EQ(source.r1_metrics.qp_predicted_cost_reduction, 101.0);
  EXPECT_EQ(source.trajectory.size(), 3u);
  EXPECT_EQ(source.format_version,
      sdu_apex::qp_fixture::kFailureContextVersion);
  EXPECT_FLOAT_EQ(source.model.weight_source_corridor_reserve,12345.f);
  EXPECT_FLOAT_EQ(source.model.weight_source_lateral_slip,500.f);
  EXPECT_EQ(source.objective.successor_recourse_valid,1);
  EXPECT_EQ(source.objective.successor_recourse_horizon,8);
  EXPECT_FLOAT_EQ(source.objective.successor_recourse_weight,4000.f);
  EXPECT_FLOAT_EQ(source.objective.successor_recourse_reserve_m,.04f);
  EXPECT_FLOAT_EQ(source.objective.successor_recourse_dt,.025f);
  EXPECT_FLOAT_EQ(source.objective.successor_recourse_deltas[3].throttle_command,-.07f);
  // v1-v3 objective had no successor policy. Reconstruct the old binary
  // layout by physically removing the newly appended schedule fields,
  // rather than only relabeling a v4 file's header.
  const auto context_path=entry / "source_context.mctx";
  std::ifstream original(context_path,std::ios::binary);
  ASSERT_TRUE(original.is_open());
  std::vector<char> full((std::istreambuf_iterator<char>(original)),
                          std::istreambuf_iterator<char>());
  constexpr size_t header_size=8+7*sizeof(std::uint32_t);
  const size_t schedule_offset=header_size+
      sizeof(std::int64_t)+sizeof(double)*2+sizeof(float)+sizeof(int)*2+
      sizeof(MpcRtiState_t)+sizeof(MpcRtiConfiguration_t);
  const size_t old_objective_size=
      offsetof(MpcRtiObjectiveSchedule_t,successor_recourse_valid);
  ASSERT_GE(full.size(),schedule_offset+sizeof(MpcRtiObjectiveSchedule_t));
  full.erase(full.begin()+schedule_offset+old_objective_size,
             full.begin()+schedule_offset+sizeof(MpcRtiObjectiveSchedule_t));
  const std::uint32_t legacy_size=old_objective_size;
  std::memcpy(full.data()+8+3*sizeof(std::uint32_t),
              &legacy_size,sizeof(legacy_size));
  for(const std::uint32_t version:{3u,2u}) {
    std::memcpy(full.data()+8,&version,sizeof(version));
    std::ofstream legacy(context_path,std::ios::binary|std::ios::trunc);
    ASSERT_TRUE(legacy.is_open());
    legacy.write(full.data(),full.size());legacy.close();
    sdu_apex::qp_fixture::FailureReplayContext migrated{};
    ASSERT_TRUE(sdu_apex::qp_fixture::read_failure_context(
        context_path.string(),&migrated)) << "version=" << version;
    EXPECT_EQ(migrated.format_version,version);
    EXPECT_EQ(migrated.objective.successor_recourse_valid,0);
    EXPECT_FLOAT_EQ(migrated.objective.successor_recourse_weight,0.f);
    EXPECT_FLOAT_EQ(migrated.model.weight_source_corridor_reserve,
        version==2?0.f:12345.f);
    EXPECT_FLOAT_EQ(migrated.model.active_speed_ceiling_mps,5.f);
    EXPECT_DOUBLE_EQ(migrated.initial_state.plant.u,4.25f);
  }
  std::filesystem::remove_all(base);
}

TEST(MpcQpFixture, V6RetainsThreeDimensionalHardProgressPlanes)
{
  char path[]="/tmp/sdu_mpc_progress_qp_XXXXXX";
  const int fd=mkstemp(path);
  ASSERT_GE(fd,0);close(fd);
  struct Cleanup {const char* path;~Cleanup(){std::remove(path);}} cleanup{path};
  MpcRtiQpCapture_t capture{};
  capture.valid=capture.capture_count=1;
  capture.problem.horizon=RICCATI_HORIZON;
  auto &polygon=capture.solver_config.state_polygons[1];
  polygon.count=1;polygon.normal[0][0]=1.f;
  polygon.progress_normal[0]=.0325f;polygon.upper[0]=.25f;
  capture.solver_config.rho=capture.solver_config.rho_u=7.f;
  capture.solver_config.rho_max=32.f;
  capture.solver_config.tolerance=.01f;
  capture.solver_config.max_iterations=150;
  capture.solver_config.adaptive_rho_update_period=10;
  capture.solver_config.over_relaxation=1.6f;
  std::fill_n(capture.solver_config.state_scale,RICCATI_MAX_NX,1.f);
  std::fill_n(capture.solver_config.input_scale,RICCATI_MAX_NU,1.f);
  ASSERT_TRUE(sdu_apex::qp_fixture::write_capture(path,capture));
  std::ifstream stream(path,std::ios::binary);
  char magic[8]{};
  stream.read(magic,sizeof(magic));
  EXPECT_EQ(std::string(magic,sizeof(magic)),"SDUFQP6\n");
  MpcRtiQpCapture_t read{};
  ASSERT_TRUE(sdu_apex::qp_fixture::read_capture(path,&read));
  EXPECT_FLOAT_EQ(read.solver_config.state_polygons[1].progress_normal[0],.0325f);
  EXPECT_FLOAT_EQ(read.problem.state_polygons[1].progress_normal[0],.0325f);
}

TEST(MpcQpFixture, V7ElasticCorridorUsesLosslessProgressAndSlackMetadata)
{
  char path[]="/tmp/sdu_mpc_elastic_qp_XXXXXX";
  const int fd=mkstemp(path);
  ASSERT_GE(fd,0);close(fd);
  struct Cleanup {const char* path;~Cleanup(){std::remove(path);}} cleanup{path};
  auto capture=std::make_unique<MpcRtiQpCapture_t>();
  capture->valid=capture->capture_count=1;
  capture->problem.horizon=RICCATI_HORIZON;
  auto &polygon=capture->solver_config.state_polygons[3];
  polygon.count=2;
  polygon.normal[0][0]=1.f;polygon.normal[1][1]=1.f;
  polygon.progress_normal[0]=.11f;
  polygon.upper[0]=.25f;polygon.upper[1]=.2f;
  polygon.elastic_weight=8000.f;
  capture->solver_config.rho=capture->solver_config.rho_u=7.f;
  capture->solver_config.rho_max=32.f;
  capture->solver_config.tolerance=.01f;
  capture->solver_config.max_iterations=300;
  capture->solver_config.adaptive_rho_update_period=10;
  capture->solver_config.over_relaxation=1.6f;
  std::fill_n(capture->solver_config.state_scale,RICCATI_MAX_NX,1.f);
  std::fill_n(capture->solver_config.input_scale,RICCATI_MAX_NU,1.f);
  ASSERT_TRUE(sdu_apex::qp_fixture::write_capture(path,*capture));
  std::ifstream stream(path,std::ios::binary);
  char magic[8]{};
  stream.read(magic,sizeof(magic));
  EXPECT_EQ(std::string(magic,sizeof(magic)),"SDUFQP7\n");
  auto read=std::make_unique<MpcRtiQpCapture_t>();
  ASSERT_TRUE(sdu_apex::qp_fixture::read_capture(path,read.get()));
  EXPECT_FLOAT_EQ(read->solver_config.state_polygons[3].progress_normal[0],.11f);
  EXPECT_FLOAT_EQ(read->solver_config.state_polygons[3].elastic_weight,8000.f);
  EXPECT_FLOAT_EQ(read->problem.state_polygons[3].elastic_weight,8000.f);
  EXPECT_FLOAT_EQ(read->problem.state_polygons[3].upper[1],.2f);
}

TEST(MpcQpFixture, ElasticFullHorizonRiccatiMatchesExplicitSlackOracleFixture)
{
  // Reconstructible 30-stage constrained LTV QP in the same 23-state/two-
  // input dimensions as the race solver. Two incompatible ORIGINAL walls
  // make the hard physical subproblem infeasible on purpose. Native ADMM
  // must solve their explicit quadratic-slack relaxation; the independent
  // Python/OSQP oracle reads the emitted V7 fixture and introduces a separate
  // nonnegative slack variable for every wall at every stage.
  auto capture=std::make_unique<MpcRtiQpCapture_t>();
  auto &qp=capture->problem;
  auto &cfg=capture->solver_config;
  qp.horizon=RICCATI_HORIZON;
  qp.x0[MPC_OPT_IDX_EY]=.6f;
  qp.x0[MPC_OPT_IDX_EPSI]=.12f;
  qp.x0[MPC_OPT_IDX_PROGRESS_DEVIATION]=.20f;
  for(int k=0;k<RICCATI_HORIZON;++k) {
    auto &step=qp.steps[k];
    for(int i=0;i<RICCATI_MAX_NX;++i) {
      step.A[i][i]=1.f;
      step.Q_diag[i]=.01f;
      step.x_lb[i]=-BIG_BOUND;
      step.x_ub[i]=BIG_BOUND;
    }
    step.Q_diag[MPC_OPT_IDX_EY]=1.f;
    step.Q_diag[MPC_OPT_IDX_EPSI]=.5f;
    step.Q_diag[MPC_OPT_IDX_PROGRESS_DEVIATION]=.1f;
    step.B[MPC_OPT_IDX_EY][0]=.20f;
    step.B[MPC_OPT_IDX_EPSI][1]=.1f;
    step.B[MPC_OPT_IDX_PROGRESS_DEVIATION][1]=.2f;
    step.R_diag[0]=.15f;
    step.R_diag[1]=.25f;
    step.u_lb[0]=-.3f;
    step.u_ub[0]=.3f;
    step.u_lb[1]=-.4f;
    step.u_ub[1]=.4f;
  }
  for(int i=0;i<RICCATI_MAX_NX;++i) {
    qp.terminal_Q[i]=.01f;
    qp.terminal_x_lb[i]=-BIG_BOUND;
    qp.terminal_x_ub[i]= BIG_BOUND;
  }
  qp.terminal_Q[MPC_OPT_IDX_EY]=1.f;
  cfg.rho=cfg.rho_u=7.f;
  cfg.rho_max=32.f;
  cfg.tolerance=.00001f;
  cfg.max_iterations=300;
  cfg.adaptive_rho_update_period=10;
  cfg.over_relaxation=1.6f;
  for(int i=0;i<RICCATI_MAX_NX;++i)cfg.state_scale[i]=1.f;
  for(int i=0;i<RICCATI_MAX_NU;++i)cfg.input_scale[i]=1.f;
  for(int k=1;k<=RICCATI_HORIZON;++k) {
    auto &polygon=cfg.state_polygons[k];
    polygon.count=3;
    polygon.normal[0][0]=1.f;
    polygon.upper[0]=.2f;
    polygon.normal[1][0]=-1.f;
    polygon.upper[1]=-.26f;
    polygon.normal[2][0]=1.f;
    polygon.normal[2][1]=.3f;
    polygon.progress_normal[2]=.08f;
    polygon.upper[2]=.24f;
    polygon.elastic_weight=30.f;
    qp.state_polygons[k]=polygon;
  }
  capture->valid=capture->capture_count=1;
  RiccatiAdmmState_t warm{};
  riccati_admm_state_init(&warm);
  capture->initial_solver_state=warm;
  RiccatiSolution_t solved{};
  float raw_u[RICCATI_HORIZON][RICCATI_MAX_NU]{};
  const auto status=riccati_admm_solve_with_primal_controls(
      qp.steps,qp.terminal_Q,qp.terminal_q,qp.terminal_x_lb,
      qp.terminal_x_ub,qp.x0,&cfg,&warm,&solved,raw_u);
  capture->raw_status=status;
  capture->raw_solution=solved;
  capture->final_solver_state=warm;
  for(int k=0;k<RICCATI_HORIZON;++k)
    for(int j=0;j<RICCATI_MAX_NU;++j)
      capture->raw_primal_u[k][j]=raw_u[k][j];
  ASSERT_TRUE(sdu_apex::qp_fixture::write_capture(
      "/tmp/mpc_elastic_horizon_parity_20261103.fqp",*capture));
  {
    std::ofstream raw("/tmp/mpc_elastic_horizon_native_20261103.csv",
                      std::ios::trunc);
    raw.precision(12);
    for(int k=0;k<=RICCATI_HORIZON;++k) {
      raw<<"x,"<<k;
      for(int j=0;j<RICCATI_MAX_NX;++j)
        raw<<','<<solved.x[k][j];
      raw<<'\n';
    }
    for(int k=0;k<RICCATI_HORIZON;++k)
      raw<<"u,"<<k<<','<<solved.u[k][0]<<','<<solved.u[k][1]<<'\n';
  }
  std::cout<<"ELASTIC_FULL_HORIZON status="<<status
           <<" iterations="<<solved.iterations
           <<" primal="<<solved.primal_residual
           <<" dual="<<solved.dual_residual
           <<" x1="<<solved.x[1][MPC_OPT_IDX_EY]
           <<" x30="<<solved.x[30][MPC_OPT_IDX_EY]
           <<" u0="<<solved.u[0][0]<<"\n";
  EXPECT_NE(status,RICCATI_STATUS_ERROR);
  EXPECT_LT(solved.primal_residual,.02f);
  EXPECT_LT(solved.dual_residual,.02f);
  EXPECT_LT(solved.x[RICCATI_HORIZON][MPC_OPT_IDX_EY],.3f);
  EXPECT_GT(solved.x[RICCATI_HORIZON][MPC_OPT_IDX_EY],.1f);
}

#include "mpc_qp_fixture_format.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace {

std::string json_number(double value) {
  if(!std::isfinite(value))return "null";
  std::ostringstream out;out<<std::setprecision(12)<<value;return out.str();
}

float max_abs_difference(const RiccatiSolution_t &lhs,
                         const RiccatiSolution_t &rhs) {
  float maximum = 0.0f;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      maximum = std::max(maximum, std::abs(lhs.x[k][i] - rhs.x[k][i]));
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    for (int i = 0; i < RICCATI_MAX_NU; ++i)
      maximum = std::max(maximum, std::abs(lhs.u[k][i] - rhs.u[k][i]));
  return maximum;
}

void reset_duals(RiccatiAdmmState_t *state) {
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    std::fill(std::begin(state->y_x[k]), std::end(state->y_x[k]), 0.0f);
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    std::fill(std::begin(state->y_u[k]), std::end(state->y_u[k]), 0.0f);
}

void reset_penalties(RiccatiAdmmState_t *state,
                     const RiccatiAdmmConfig_t &config) {
  const auto configured_state_rho = [&config](int i) {
    return config.rho_x_by_state[i] > 0.0f
        ? config.rho_x_by_state[i] : config.rho;
  };
  const auto configured_input_rho = [&config](int i) {
    return config.rho_u_by_input[i] > 0.0f
        ? config.rho_u_by_input[i] : config.rho_u;
  };
  state->rho = config.rho;
  state->rho_u = config.rho_u;
  for (int k = 0; k <= RICCATI_HORIZON; ++k) {
    for (int i = 0; i < RICCATI_MAX_NX; ++i) {
      const float previous = state->rho_state[k][i] > 0.0f
          ? state->rho_state[k][i] : state->rho;
      const float rho = configured_state_rho(i);
      if (state->initialized && previous > 0.0f && rho > 0.0f &&
          previous != rho)
        state->y_x[k][i] *= previous / rho;
      state->rho_state[k][i] = rho;
    }
  }
  for (int k = 0; k < RICCATI_HORIZON; ++k) {
    for (int i = 0; i < RICCATI_MAX_NU; ++i) {
      const float previous = state->rho_input[k][i] > 0.0f
          ? state->rho_input[k][i] : state->rho_u;
      const float rho = configured_input_rho(i);
      if (state->initialized && previous > 0.0f && rho > 0.0f &&
          previous != rho)
        state->y_u[k][i] *= previous / rho;
      state->rho_input[k][i] = rho;
    }
  }
}

void select_seed(const std::string &mode, const MpcRtiQpCapture_t &capture,
                 RiccatiAdmmConfig_t *config,
                 RiccatiAdmmState_t *state) {
  *state = capture.initial_solver_state;
  if (mode == "recorded") return;
  if (mode == "cold") {
    riccati_admm_state_init(state);
    return;
  }
  if (mode == "dual-reset") {
    reset_duals(state);
    return;
  }
  if (mode == "primal-only") {
    reset_duals(state);
    reset_penalties(state, *config);
    state->initialized = 1;
    return;
  }
  if (mode == "penalty-reset") {
    reset_penalties(state, *config);
    return;
  }
  throw std::runtime_error("seed mode must be recorded, cold, primal-only, "
                           "dual-reset, or penalty-reset");
}

void select_rho_policy(const std::string &mode,
                       RiccatiAdmmConfig_t *config) {
  if (mode == "recorded") return;
  if (mode == "fixed") {
    config->adaptive_rho = 0;
    return;
  }
  if (mode == "adaptive") {
    config->adaptive_rho = 1;
    return;
  }
  throw std::runtime_error("rho policy must be recorded, fixed, or adaptive");
}

struct Metrics {
  double primal_objective = 0.0;
  double projected_objective = 0.0;
  float max_primal_dynamics_residual = 0.0f;
  float max_projected_dynamics_residual = 0.0f;
  float max_primal_state_bound_violation = 0.0f;
  float max_projected_state_bound_violation = 0.0f;
  float max_primal_input_bound_violation = 0.0f;
  float max_projected_input_bound_violation = 0.0f;
  float max_primal_projected_u_delta = 0.0f;
  float max_primal_polygon_violation_m = 0.f;
  float max_projected_polygon_violation_m = 0.f;
  int maximum_polygon_primal_stage = -1;
  int elastic_polygon_stages = 0;
  double primal_elastic_slack_cost = 0.;
  double projected_elastic_slack_cost = 0.;
};

Metrics evaluate(const MpcRtiProblem_t &problem,
                 const RiccatiSolution_t &solution,
                 const float primal_u[][RICCATI_MAX_NU],
                 const RiccatiAdmmState_t &final_state) {
  Metrics result;
  float projected_x[RICCATI_HORIZON + 1][RICCATI_MAX_NX];
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      projected_x[k][i] = final_state.z_x[k][i] *
          (final_state.scaling_enabled ? final_state.state_scale[i] : 1.0f);

  // The polygon (not the independent e_y box) is the actual active
  // footprint envelope. x/z agreement alone is insufficient for claiming
  // that a frozen QP's raw physical trajectory is inside that envelope.
  for(int k=1;k<=RICCATI_HORIZON;++k) {
    const auto &polygon=problem.state_polygons[k];
    if(polygon.elastic_weight>0.f)++result.elastic_polygon_stages;
    for(int f=0;f<polygon.count;++f) {
      const float *n=polygon.normal[f];
      const float progress=polygon.progress_normal[f];
      const float upper=polygon.upper[f];
      const float raw=n[0]*solution.x[k][MPC_OPT_IDX_EY]+
                      n[1]*solution.x[k][MPC_OPT_IDX_EPSI]+
                      progress*solution.x[k][MPC_OPT_IDX_PROGRESS_DEVIATION]-upper;
      const float projected=n[0]*projected_x[k][MPC_OPT_IDX_EY]+
                            n[1]*projected_x[k][MPC_OPT_IDX_EPSI]+
                            progress*projected_x[k][MPC_OPT_IDX_PROGRESS_DEVIATION]-upper;
      if(raw>result.max_primal_polygon_violation_m) {
        result.max_primal_polygon_violation_m=raw;
        result.maximum_polygon_primal_stage=k;
      }
      result.max_projected_polygon_violation_m=std::max(
         result.max_projected_polygon_violation_m,projected);
      if(polygon.elastic_weight>0.f) {
        // The QP remains convex in the ORIGINAL wall plane, with an
        // explicitly penalized xi=max(0,a*x-b). A positive xi is not
        // hard QP infeasibility but it is NOT a legal source trajectory.
        const double xi=std::max(0.,double(raw));
        const double zi=std::max(0.,double(projected));
        result.primal_elastic_slack_cost+=.5*polygon.elastic_weight*xi*xi;
        result.projected_elastic_slack_cost+=.5*polygon.elastic_weight*zi*zi;
      }
    }
  }
  for (int k = 0; k < problem.horizon; ++k) {
    const auto &step = problem.steps[k];
    for (int i = 0; i < RICCATI_MAX_NX; ++i) {
      double primal_predicted = step.d[i];
      double projected_predicted = step.d[i];
      for (int j = 0; j < RICCATI_MAX_NX; ++j)
      {
        primal_predicted += static_cast<double>(step.A[i][j]) *
            solution.x[k][j];
        projected_predicted += static_cast<double>(step.A[i][j]) *
            projected_x[k][j];
      }
      for (int j = 0; j < RICCATI_MAX_NU; ++j)
      {
        const float z_u = final_state.z_u[k][j] *
            (final_state.scaling_enabled
                ? final_state.input_scale[j] : 1.0f);
        primal_predicted += static_cast<double>(step.B[i][j]) * primal_u[k][j];
        projected_predicted += static_cast<double>(step.B[i][j]) * z_u;
      }
      result.max_primal_dynamics_residual = std::max(
          result.max_primal_dynamics_residual,
          static_cast<float>(std::abs(solution.x[k + 1][i] -
                                      primal_predicted)));
      result.max_projected_dynamics_residual = std::max(
          result.max_projected_dynamics_residual,
          static_cast<float>(std::abs(projected_x[k + 1][i] -
                                      projected_predicted)));
      if (k > 0) {
        const float x_primal = solution.x[k][i];
        const float x_projected = projected_x[k][i];
        // +/-BIG_BOUND means unconstrained in the ADMM solver, NOT a
        // physical +/-50-unit box. Wheel angular velocities can routinely
        // exceed 50 rad/s. Report only the violations that the QP enforces.
        const auto box_violation = [&](float state) {
          const float lower = step.x_lb[i] > -BIG_BOUND
              ? step.x_lb[i] - state : 0.0f;
          const float upper = step.x_ub[i] < BIG_BOUND
              ? state - step.x_ub[i] : 0.0f;
          return std::max(0.0f, std::max(lower, upper));
        };
        result.max_primal_state_bound_violation = std::max(
            result.max_primal_state_bound_violation,
            box_violation(x_primal));
        result.max_projected_state_bound_violation = std::max(
            result.max_projected_state_bound_violation,
            box_violation(x_projected));
      }
      const float x_primal = solution.x[k][i];
      const float x_projected = projected_x[k][i];
      result.primal_objective +=
          0.5 * step.Q_diag[i] * x_primal * x_primal + step.q[i] * x_primal;
      result.projected_objective +=
          0.5 * step.Q_diag[i] * x_projected * x_projected +
          step.q[i] * x_projected;
      for (int j = 0; j < RICCATI_MAX_NX; ++j) {
        result.primal_objective += 0.5 * step.Q_offdiag[i][j] *
            x_primal * solution.x[k][j];
        result.projected_objective += 0.5 * step.Q_offdiag[i][j] *
            x_projected * projected_x[k][j];
      }
    }
    for (int i = 0; i < RICCATI_MAX_NU; ++i) {
      const float u_primal = primal_u[k][i];
      const float u_projected = final_state.z_u[k][i] *
          (final_state.scaling_enabled
              ? final_state.input_scale[i] : 1.0f);
      result.max_primal_input_bound_violation = std::max(
          result.max_primal_input_bound_violation,
          std::max(0.0f, std::max(step.u_lb[i] - u_primal,
                                  u_primal - step.u_ub[i])));
      result.max_projected_input_bound_violation = std::max(
          result.max_projected_input_bound_violation,
          std::max(0.0f, std::max(step.u_lb[i] - u_projected,
                                  u_projected - step.u_ub[i])));
      result.max_primal_projected_u_delta = std::max(
          result.max_primal_projected_u_delta,
          std::abs(u_primal - u_projected));
      result.primal_objective +=
          0.5 * step.R_diag[i] * u_primal * u_primal + step.r[i] * u_primal;
      result.projected_objective +=
          0.5 * step.R_diag[i] * u_projected * u_projected +
          step.r[i] * u_projected;
      if (i == 0) {
        result.primal_objective += step.R_cross *
            u_primal * primal_u[k][1];
        result.projected_objective += step.R_cross *
            u_projected * final_state.z_u[k][1] *
            (final_state.scaling_enabled ? final_state.input_scale[1] : 1.f);
      }
      for (int j = 0; j < RICCATI_MAX_NX; ++j) {
        result.primal_objective +=
            static_cast<double>(step.N[j][i]) * solution.x[k][j] * u_primal;
        result.projected_objective +=
            static_cast<double>(step.N[j][i]) * projected_x[k][j] * u_projected;
      }
    }
  }
  for (int i = 0; i < RICCATI_MAX_NX; ++i) {
    const float x_primal = solution.x[problem.horizon][i];
    const float x_projected = projected_x[problem.horizon][i];
    const auto terminal_box_violation = [&](float state) {
      const float lower = problem.terminal_x_lb[i] > -BIG_BOUND
          ? problem.terminal_x_lb[i] - state : 0.0f;
      const float upper = problem.terminal_x_ub[i] < BIG_BOUND
          ? state - problem.terminal_x_ub[i] : 0.0f;
      return std::max(0.0f, std::max(lower, upper));
    };
    result.max_primal_state_bound_violation = std::max(
        result.max_primal_state_bound_violation,
        terminal_box_violation(x_primal));
    result.max_projected_state_bound_violation = std::max(
        result.max_projected_state_bound_violation,
        terminal_box_violation(x_projected));
    result.primal_objective +=
        0.5 * problem.terminal_Q[i] * x_primal * x_primal +
        problem.terminal_q[i] * x_primal;
    result.projected_objective +=
        0.5 * problem.terminal_Q[i] * x_projected * x_projected +
        problem.terminal_q[i] * x_projected;
  }
  result.primal_objective+=result.primal_elastic_slack_cost;
  result.projected_objective+=result.projected_elastic_slack_cost;
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 4 && std::string(argv[1]) == "--roundtrip") {
    MpcRtiQpCapture_t capture{};
    if (!sdu_apex::qp_fixture::read_capture(argv[2], &capture) ||
        !sdu_apex::qp_fixture::write_capture(argv[3], capture)) {
      std::cerr << "mpc_qp_fixture_replay: fixture roundtrip failed\n";
      return 2;
    }
    std::cout << "roundtrip_ok\n";
    return 0;
  }
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: mpc_qp_fixture_replay FIXTURE.fqp "
                 "recorded|cold|primal-only|dual-reset|penalty-reset "
                 "[recorded|fixed|adaptive]\n"
                 "       mpc_qp_fixture_replay --roundtrip INPUT.fqp OUTPUT.fqp\n";
    return 2;
  }
  try {
    MpcRtiQpCapture_t capture{};
    if (!sdu_apex::qp_fixture::read_capture(argv[1], &capture))
      throw std::runtime_error("fixture failed schema, bounds, or EOF checks");
    // Diagnostic only: test whether the ORIGINAL wall polygons of an
    // elastic frozen QP are actually feasible for the same affine model,
    // same actuator bounds and same racing Hessian. Never change live QP.
    if(const char *hard=std::getenv("MPC_QP_BENCH_HARD_ORIGINAL_WALLS")) {
      if(std::string(hard)!="1")
        throw std::runtime_error("hard-wall diagnostic must be 1");
      for(int k=1;k<=RICCATI_HORIZON;++k) {
        capture.solver_config.state_polygons[k].elastic_weight=0.f;
        capture.problem.state_polygons[k].elastic_weight=0.f;
      }
    }
    RiccatiAdmmConfig_t config = capture.solver_config;
    RiccatiAdmmState_t state{};
    select_seed(argv[2], capture, &config, &state);
    const std::string rho_policy = argc == 4 ? argv[3] : "recorded";
    select_rho_policy(rho_policy, &config);
    // Isolated diagnostic parameter sweeps; this does NOT modify the
    // production ROS MPC profile, physical plant, or original QP objective.
    // Reinitialize the ADMM state when penalties change, so adapted warm
    // multipliers cannot masquerade as a better cold-start policy.
    const auto numeric_env = [](const char *name, float old_value) {
      const char *text = std::getenv(name);
      if (!text) return old_value;
      const double parsed = std::strtod(text, nullptr);
      if (!std::isfinite(parsed) || parsed <= 0.)
        throw std::runtime_error(std::string("invalid ") + name);
      return static_cast<float>(parsed);
    };
    config.rho = numeric_env("MPC_QP_BENCH_RHO", config.rho);
    config.rho_u = numeric_env("MPC_QP_BENCH_RHO_U", config.rho_u);
    config.rho_max = numeric_env("MPC_QP_BENCH_RHO_MAX", config.rho_max);
    config.over_relaxation = numeric_env(
        "MPC_QP_BENCH_OVER_RELAXATION",config.over_relaxation);
    config.tolerance = numeric_env(
        "MPC_QP_BENCH_TOLERANCE",config.tolerance);
    if (config.rho_max > 127.f || config.over_relaxation > 2.f)
      throw std::runtime_error("invalid bench penalty or relaxation");
    if (const char *iter = std::getenv("MPC_QP_BENCH_MAX_ITER")) {
      const int count = std::atoi(iter);
      if (count <= 0 || count > 10000)
        throw std::runtime_error("invalid MPC_QP_BENCH_MAX_ITER");
      config.max_iterations = count;
    }
    if (const char *unbounded = std::getenv("MPC_QP_BENCH_UNBOUNDED")) {
      if (std::string(unbounded) != "1")
        throw std::runtime_error("MPC_QP_BENCH_UNBOUNDED must be 1 when set");
      config.max_solve_time_us = 0.0;
    }
    if (std::getenv("MPC_QP_BENCH_STATE_RHO")) {
      const float value=numeric_env("MPC_QP_BENCH_STATE_RHO",0.f);
      if (value < 1.f || value > 127.f)
        throw std::runtime_error("invalid MPC_QP_BENCH_STATE_RHO");
      for (auto &entry:config.rho_x_by_state) entry=value;
    }
    if (std::getenv("MPC_QP_BENCH_INPUT_RHO")) {
      const float value=numeric_env("MPC_QP_BENCH_INPUT_RHO",0.f);
      if (value > 127.f)
        throw std::runtime_error("invalid MPC_QP_BENCH_INPUT_RHO");
      for (auto &entry:config.rho_u_by_input) entry=value;
    }
    if (const char *coordinate = std::getenv(
            "MPC_QP_BENCH_STATE_COORDINATE")) {
      const int index = std::atoi(coordinate);
      const float value = numeric_env("MPC_QP_BENCH_COORDINATE_RHO",0.f);
      if (index < 0 || index >= RICCATI_MAX_NX || value < 1.f ||
          value > 127.f)
        throw std::runtime_error("invalid bench state coordinate/rho");
      config.rho_x_by_state[index]=value;
      riccati_admm_state_init(&state);
    }
    if (std::getenv("MPC_QP_BENCH_RHO") ||
        std::getenv("MPC_QP_BENCH_RHO_U") ||
        std::getenv("MPC_QP_BENCH_STATE_RHO") ||
        std::getenv("MPC_QP_BENCH_INPUT_RHO") ||
        std::getenv("MPC_QP_BENCH_RHO_MAX"))
      riccati_admm_state_init(&state);
    if (std::getenv("MPC_QP_BENCH_NO_SCALING")) {
      config.use_scaling=0;
      riccati_admm_state_init(&state);
    }
    const char *period_text = std::getenv("MPC_QP_BENCH_RHO_PERIOD");
    const int bench_period = period_text ? std::atoi(period_text) : 0;
    if (bench_period > 0 && bench_period <= 150)
      config.adaptive_rho_update_period = bench_period;
    RiccatiSolution_t solution{};
    float primal_u[RICCATI_HORIZON][RICCATI_MAX_NU]{};
    const RiccatiStatus_t status = riccati_admm_solve_with_primal_controls(
        capture.problem.steps, capture.problem.terminal_Q,
        capture.problem.terminal_q, capture.problem.terminal_x_lb,
        capture.problem.terminal_x_ub, capture.problem.x0, &config, &state,
        &solution, primal_u);
    // Opt-in reproducible solver-kernel microbenchmark for recorded QPs:
    // reset ADMM warm state for each measurement; do not include fixture
    // I/O or startup in solve duration. Parity check below still applies to
    // the original recorded single solve.
    const char *bench_text = std::getenv("MPC_QP_BENCH_REPEATS");
    const int bench_repeats = bench_text ? std::max(0, std::min(1000,
        std::atoi(bench_text))) : 0;
    std::vector<double> bench_us;
    bench_us.reserve(bench_repeats);
    for (int rep = 0; rep < bench_repeats; ++rep) {
      RiccatiAdmmState_t test_state{};
      RiccatiAdmmConfig_t test_config = capture.solver_config;
      select_seed(argv[2], capture, &test_config, &test_state);
      select_rho_policy(rho_policy, &test_config);
      // A benchmark with overridden penalties should use a single solve,
      // unless an explicit fixed benchmark fixture is constructed instead.
      if (std::getenv("MPC_QP_BENCH_RHO") ||
          std::getenv("MPC_QP_BENCH_RHO_U") ||
          std::getenv("MPC_QP_BENCH_STATE_RHO") ||
          std::getenv("MPC_QP_BENCH_INPUT_RHO"))
        throw std::runtime_error(
            "benchmark repeats not supported with penalty overrides");
      if (bench_period > 0 && bench_period <= 150)
        test_config.adaptive_rho_update_period = bench_period;
      RiccatiSolution_t test_solution{};
      float test_u[RICCATI_HORIZON][RICCATI_MAX_NU]{};
      const auto begin = std::chrono::steady_clock::now();
      const RiccatiStatus_t test_status =
          riccati_admm_solve_with_primal_controls(
              capture.problem.steps, capture.problem.terminal_Q,
              capture.problem.terminal_q, capture.problem.terminal_x_lb,
              capture.problem.terminal_x_ub, capture.problem.x0, &test_config,
              &test_state, &test_solution, test_u);
      const auto end = std::chrono::steady_clock::now();
      if (test_status != status || test_solution.iterations != solution.iterations ||
          max_abs_difference(test_solution, solution) > 1.e-6f)
        throw std::runtime_error("benchmark solve differed from parity baseline");
      const auto microseconds =
          std::chrono::duration<double, std::micro>(end-begin).count();
      bench_us.push_back(microseconds);
    }
    std::sort(bench_us.begin(), bench_us.end());
    const auto bench_at = [&bench_us](double quantile) -> double {
      if (bench_us.empty()) return 0.;
      return bench_us[std::min(bench_us.size()-1, static_cast<std::size_t>(
          quantile*(bench_us.size()-1)))];
    };
    RiccatiDebugInfo_t profile{};
    riccati_debug_get_last(&profile);
    // Diagnostic-only native raw decision output for independent OSQP
    // vector parity. Never read by the controller or used for publication.
    if(const char *out=std::getenv("MPC_QP_BENCH_RAW_SOLUTION_PATH")) {
      std::ofstream stream(out);
      if(!stream.good())throw std::runtime_error("raw QP export failed");
      stream<<std::setprecision(15);
      for(int k=0;k<=RICCATI_HORIZON;++k) {
        for(int i=0;i<RICCATI_MAX_NX;++i)
          stream<<solution.x[k][i]<<(i+1==RICCATI_MAX_NX?'\n':' ');
      }
      for(int k=0;k<RICCATI_HORIZON;++k)
        stream<<solution.u[k][0]<<" "<<solution.u[k][1]<<'\n';
    }
    const Metrics metrics = evaluate(capture.problem, solution,
        primal_u, state);
    const float parity_delta = max_abs_difference(solution,
                                                   capture.raw_solution);
    const bool parity = status == capture.raw_status &&
        solution.status == capture.raw_solution.status &&
        solution.iterations == capture.raw_solution.iterations &&
        parity_delta <= 1.0e-6f;
    std::cout << "{\"schema\":\"sdu-apex-frozen-qp-replay-v1\","
              << "\"regime\":\"identical_qp\","
              << "\"scaling_enabled\":" << config.use_scaling << ','
              << "\"seed_mode\":\"" << argv[2] << "\","
              << "\"rho_policy\":\"" << rho_policy << "\","
              << "\"status\":" << static_cast<int>(status) << ','
              << "\"iterations\":" << solution.iterations << ','
              << "\"primal_residual\":" << json_number(solution.primal_residual) << ','
              << "\"dual_residual\":" << json_number(solution.dual_residual) << ','
              << "\"primal_objective\":" << json_number(metrics.primal_objective) << ','
              << "\"projected_objective\":" << json_number(metrics.projected_objective) << ','
              << "\"max_primal_dynamics_equality_residual\":"
              << json_number(metrics.max_primal_dynamics_residual) << ','
              << "\"max_projected_dynamics_equality_residual\":"
              << json_number(metrics.max_projected_dynamics_residual) << ','
              << "\"max_primal_state_bound_violation\":"
              << json_number(metrics.max_primal_state_bound_violation) << ','
              << "\"max_projected_state_bound_violation\":"
              << json_number(metrics.max_projected_state_bound_violation) << ','
              << "\"max_primal_input_bound_violation\":"
              << json_number(metrics.max_primal_input_bound_violation) << ','
              << "\"max_projected_input_bound_violation\":"
              << json_number(metrics.max_projected_input_bound_violation) << ','
              << "\"elastic_polygon_stages\":"
              << metrics.elastic_polygon_stages << ','
              << "\"primal_elastic_slack_cost\":"
              << json_number(metrics.primal_elastic_slack_cost) << ','
              << "\"projected_elastic_slack_cost\":"
              << json_number(metrics.projected_elastic_slack_cost) << ','
              << "\"max_primal_polygon_violation_m\":"
              << json_number(metrics.max_primal_polygon_violation_m) << ','
              << "\"max_projected_polygon_violation_m\":"
              << json_number(metrics.max_projected_polygon_violation_m) << ','
              << "\"maximum_polygon_primal_stage\":"
              << metrics.maximum_polygon_primal_stage << ','
              << "\"max_primal_vs_projected_input\":"
              << json_number(metrics.max_primal_projected_u_delta) << ','
              << "\"recorded_result_max_abs_delta\":" << json_number(parity_delta) << ','
              << "\"factorization_count\":" << profile.quadratic_factorization_count << ','
              << "\"factorization_ns\":" << profile.quadratic_factorization_time_ns << ','
              << "\"rhs_ns\":" << profile.quadratic_rhs_time_ns << ','
              << "\"projection_ns\":" << profile.projection_residual_time_ns << ','
              << "\"rho_update_ns\":" << profile.rho_update_time_ns << ','
              << "\"scaling_ns\":" << profile.scaling_transform_time_ns << ','
              << "\"output_unscale_ns\":" << profile.output_unscale_time_ns << ','
              << "\"bench_repeats\":" << bench_repeats << ','
              << "\"bench_p50_us\":" << bench_at(.50) << ','
              << "\"bench_p95_us\":" << bench_at(.95) << ','
              << "\"bench_max_us\":" << bench_at(1.) << ','
              << "\"recorded_result_exact_parity\":"
              << (parity ? "true" : "false") << "}\n";
    return status == RICCATI_STATUS_ERROR ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << "mpc_qp_fixture_replay: " << error.what() << '\n';
    return 2;
  }
}

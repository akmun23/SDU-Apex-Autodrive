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

namespace {

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
        result.max_primal_state_bound_violation = std::max(
            result.max_primal_state_bound_violation,
            std::max(0.0f, std::max(step.x_lb[i] - x_primal,
                                    x_primal - step.x_ub[i])));
        result.max_projected_state_bound_violation = std::max(
            result.max_projected_state_bound_violation,
            std::max(0.0f, std::max(step.x_lb[i] - x_projected,
                                    x_projected - step.x_ub[i])));
      }
      const float x_primal = solution.x[k][i];
      const float x_projected = projected_x[k][i];
      result.primal_objective +=
          0.5 * step.Q_diag[i] * x_primal * x_primal + step.q[i] * x_primal;
      result.projected_objective +=
          0.5 * step.Q_diag[i] * x_projected * x_projected +
          step.q[i] * x_projected;
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
      for (int j = 0; j < RICCATI_MAX_NX; ++j) {
        result.primal_objective +=
            step.N[j][i] * solution.x[k][j] * u_primal;
        result.projected_objective +=
            step.N[j][i] * projected_x[k][j] * u_projected;
      }
    }
  }
  for (int i = 0; i < RICCATI_MAX_NX; ++i) {
    const float x_primal = solution.x[problem.horizon][i];
    const float x_projected = projected_x[problem.horizon][i];
    result.max_primal_state_bound_violation = std::max(
        result.max_primal_state_bound_violation,
        std::max(0.0f, std::max(problem.terminal_x_lb[i] - x_primal,
                                x_primal - problem.terminal_x_ub[i])));
    result.max_projected_state_bound_violation = std::max(
        result.max_projected_state_bound_violation,
        std::max(0.0f, std::max(problem.terminal_x_lb[i] - x_projected,
                                x_projected - problem.terminal_x_ub[i])));
    result.primal_objective +=
        0.5 * problem.terminal_Q[i] * x_primal * x_primal +
        problem.terminal_q[i] * x_primal;
    result.projected_objective +=
        0.5 * problem.terminal_Q[i] * x_projected * x_projected +
        problem.terminal_q[i] * x_projected;
  }
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
    RiccatiAdmmConfig_t config = capture.solver_config;
    RiccatiAdmmState_t state{};
    select_seed(argv[2], capture, &config, &state);
    const std::string rho_policy = argc == 4 ? argv[3] : "recorded";
    select_rho_policy(rho_policy, &config);
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
              << "\"seed_mode\":\"" << argv[2] << "\","
              << "\"rho_policy\":\"" << rho_policy << "\","
              << "\"status\":" << static_cast<int>(status) << ','
              << "\"iterations\":" << solution.iterations << ','
              << "\"primal_residual\":" << solution.primal_residual << ','
              << "\"dual_residual\":" << solution.dual_residual << ','
              << "\"primal_objective\":" << metrics.primal_objective << ','
              << "\"projected_objective\":" << metrics.projected_objective << ','
              << "\"max_primal_dynamics_equality_residual\":"
              << metrics.max_primal_dynamics_residual << ','
              << "\"max_projected_dynamics_equality_residual\":"
              << metrics.max_projected_dynamics_residual << ','
              << "\"max_primal_state_bound_violation\":"
              << metrics.max_primal_state_bound_violation << ','
              << "\"max_projected_state_bound_violation\":"
              << metrics.max_projected_state_bound_violation << ','
              << "\"max_primal_input_bound_violation\":"
              << metrics.max_primal_input_bound_violation << ','
              << "\"max_projected_input_bound_violation\":"
              << metrics.max_projected_input_bound_violation << ','
              << "\"max_primal_vs_projected_input\":"
              << metrics.max_primal_projected_u_delta << ','
              << "\"recorded_result_max_abs_delta\":" << parity_delta << ','
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

#include "mpc_qp_fixture_format.hpp"
#include "../f1tenth_mpc/include/mpc_qp_failure_context.hpp"
#include "../f1tenth_mpc/src/mpc_rti_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double max_solution_delta(const RiccatiSolution_t &a,
                          const RiccatiSolution_t &b) {
  double delta = 0.0;
  for (int k = 0; k <= RICCATI_HORIZON; ++k)
    for (int i = 0; i < RICCATI_MAX_NX; ++i)
      delta = std::max(delta, std::abs(static_cast<double>(a.x[k][i]) - b.x[k][i]));
  for (int k = 0; k < RICCATI_HORIZON; ++k)
    for (int i = 0; i < RICCATI_MAX_NU; ++i)
      delta = std::max(delta, std::abs(static_cast<double>(a.u[k][i]) - b.u[k][i]));
  return delta;
}

const char *rollout_name(MpcRtiRolloutStatus_t status) {
  switch (status) {
    case MPC_RTI_ROLLOUT_OK: return "ok";
    case MPC_RTI_ROLLOUT_INVALID_INPUT: return "invalid_input";
    case MPC_RTI_ROLLOUT_INVALID_MODEL: return "invalid_model";
    case MPC_RTI_ROLLOUT_COMMAND_LIMIT: return "command_limit";
    case MPC_RTI_ROLLOUT_STATE_LIMIT: return "state_limit";
    case MPC_RTI_ROLLOUT_CORRIDOR: return "corridor";
    case MPC_RTI_ROLLOUT_TERMINAL_VIABILITY: return "terminal_viability";
  }
  return "unknown";
}

void json_number(double value) {
  if (std::isfinite(value)) std::cout << value;
  else std::cout << "null";
}

struct SourceScore {
  MpcRtiRolloutStatus_t original_status{MPC_RTI_ROLLOUT_INVALID_INPUT};
  int original_failure_stage{-1};
  MpcRtiRolloutStatus_t relaxed_status{MPC_RTI_ROLLOUT_INVALID_INPUT};
  int relaxed_failure_stage{-1};
  int complete{};
  double objective{NAN};
  float minimum_clearance{NAN};
};

SourceScore score_source_controls(
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcModelControl_t controls[PREDICTION_HORIZON]) {
  SourceScore result{};
  MpcRtiCycleConfiguration_t configuration{};
  configuration.model = context.model;
  configuration.model.world_sweep_check = nullptr;
  configuration.model.world_sweep_context = nullptr;
  configuration.model.world_map_rejection_only = 0;
  configuration.terminal_viability = 0;
  configuration.objective_schedule = &context.objective;

  MpcRtiNominal_t original{};
  result.original_status = mpc_rti_validate_candidate_plan(
      &context.initial_state, context.progress_m, controls,
      context.trajectory.data(), context.trajectory.size(),
      context.lap_length_m, PREDICTION_HORIZON, context.prediction_dt,
      &configuration, &original, &result.original_failure_stage);

  std::vector<MpcTrajectorySample_t> wide = context.trajectory;
  for (auto &sample : wide) sample.left_bound = sample.right_bound = 100.0;
  MpcRtiNominal_t relaxed{};
  result.relaxed_status = mpc_rti_validate_candidate_plan(
      &context.initial_state, context.progress_m, controls,
      wide.data(), wide.size(), context.lap_length_m,
      PREDICTION_HORIZON, context.prediction_dt,
      &configuration, &relaxed, &result.relaxed_failure_stage);
  if (result.relaxed_status == MPC_RTI_ROLLOUT_OK) {
    result.complete = mpc_rti_score_candidate_plan_for_diagnostics(
        &relaxed, context.trajectory.data(), context.trajectory.size(),
        context.lap_length_m, &configuration, &result.objective,
        &result.minimum_clearance);
  }
  return result;
}

void emit_restoration(const char *name, const MpcRtiQpCapture_t &capture,
                     const MpcModelControl_t *nominal_controls,
                     const sdu_apex::qp_fixture::FailureReplayContext &context) {
  MpcRtiCycleConfiguration_t validation{};
  validation.model = context.model;
  validation.model.world_sweep_check = nullptr;
  validation.model.world_sweep_context = nullptr;
  validation.model.world_map_rejection_only = 0;
  validation.terminal_viability = 0;
  validation.objective_schedule = &context.objective;
  std::vector<MpcTrajectorySample_t> wide = context.trajectory;
  for (auto &sample : wide) sample.left_bound = sample.right_bound = 100.0;
  MpcRtiNominal_t nominal{};
  int nominal_failure_stage = -1;
  const auto nominal_status = mpc_rti_validate_candidate_plan(
      &context.initial_state, context.progress_m, nominal_controls,
      wide.data(), wide.size(), context.lap_length_m,
      PREDICTION_HORIZON, context.prediction_dt, &validation,
      &nominal, &nominal_failure_stage);

  MpcRtiProblem_t problem = capture.problem;
  const bool prepared = nominal_status == MPC_RTI_ROLLOUT_OK &&
      mpc_rti_prepare_feasibility_restoration_qp(
          &problem, &nominal, 0.18f, 8000.0f);
  RiccatiStatus_t solver_status = RICCATI_STATUS_ERROR;
  RiccatiSolution_t solution{};
  MpcModelControl_t controls[PREDICTION_HORIZON]{};
  int original_failure_stage = -1;
  MpcRtiRolloutStatus_t original_status = MPC_RTI_ROLLOUT_INVALID_INPUT;
  if (prepared) {
    RiccatiAdmmConfig_t solver = capture.solver_config;
    solver.terminal_constraints = problem.terminal_constraints;
    std::copy_n(problem.state_polygons, RICCATI_HORIZON + 1,
                solver.state_polygons);
    solver.reuse_same_epoch = 0;
    solver.max_solve_time_us = 0.0;
    RiccatiAdmmState_t warm{};
    riccati_admm_state_init(&warm);
    float primal_u[RICCATI_HORIZON][RICCATI_MAX_NU]{};
    solver_status = riccati_admm_solve_with_primal_controls(
        problem.steps, problem.terminal_Q, problem.terminal_q,
        problem.terminal_x_lb, problem.terminal_x_ub, problem.x0, &solver,
        &warm, &solution, primal_u);
    float previous_steering =
        problem.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
    float previous_throttle =
        problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
    for (int k = 0; k < PREDICTION_HORIZON; ++k) {
      const auto &stage = problem.steps[k];
      const float decision_steering = std::clamp(solution.u[k][0],
          stage.u_lb[0], stage.u_ub[0]);
      const float decision_throttle = std::clamp(solution.u[k][1],
          stage.u_lb[1], stage.u_ub[1]);
      const float steering = context.model.command_increment_qp_enabled
          ? previous_steering + decision_steering : decision_steering;
      const float throttle = context.model.command_increment_qp_enabled
          ? previous_throttle + decision_throttle : decision_throttle;
      controls[k].steering_target_rad = std::clamp(steering,
          -context.model.max_steering_rad, context.model.max_steering_rad);
      controls[k].throttle_command = std::clamp(throttle,
          MPC_THROTTLE_MIN_DRIVE, MPC_THROTTLE_MAX);
      previous_steering = controls[k].steering_target_rad;
      previous_throttle = controls[k].throttle_command;
    }
    MpcRtiNominal_t validated{};
    original_status = mpc_rti_validate_candidate_plan(
        &context.initial_state, context.progress_m, controls,
        context.trajectory.data(), context.trajectory.size(),
        context.lap_length_m, PREDICTION_HORIZON, context.prediction_dt,
        &validation, &validated, &original_failure_stage);
  }
  const SourceScore restored = prepared
      ? score_source_controls(context, controls) : SourceScore{};
  std::cout << "{\"pass\":\"" << name
            << "_secondary_restoration_probe\",\"offline_probe\":true"
            << ",\"nominal_rollout_status\":\""
            << rollout_name(nominal_status)
            << "\",\"qp_prepared\":" << (prepared ? "true" : "false")
            << ",\"qp_status\":" << static_cast<int>(solver_status)
            << ",\"iterations\":" << solution.iterations
            << ",\"primal_residual\":";
  json_number(solution.primal_residual);
  std::cout << ",\"dual_residual\":";
  json_number(solution.dual_residual);
  std::cout << ",\"source_corridor_status\":\""
            << rollout_name(original_status)
            << "\",\"source_corridor_failure_stage\":"
            << original_failure_stage
            << ",\"relaxed_source_status\":\""
            << rollout_name(restored.relaxed_status)
            << "\",\"source_objective\":";
  json_number(restored.objective);
  std::cout << ",\"source_minimum_clearance_m\":";
  json_number(restored.minimum_clearance);
  std::cout << "}\n";
}

void emit_first_stage_recoverability(
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcModelControl_t *seed_controls) {
  MpcRtiCycleConfiguration_t validation{};
  validation.model = context.model;
  validation.model.world_sweep_check = nullptr;
  validation.model.world_sweep_context = nullptr;
  validation.model.world_map_rejection_only = 0;
  validation.terminal_viability = 0;
  const auto &plant = context.initial_state.plant;
  const double max_steering_step = context.model.max_steering_command_step_rad > 0.f
      ? context.model.max_steering_command_step_rad
      : 2.0 * context.model.max_steering_rad;
  const double max_throttle_step = context.model.max_throttle_command_step > 0.f
      ? context.model.max_throttle_command_step : MPC_THROTTLE_MAX;
  const double steering_lo = std::max(-static_cast<double>(context.model.max_steering_rad),
      static_cast<double>(plant.delayed_steering_command_2) - max_steering_step);
  const double steering_hi = std::min(static_cast<double>(context.model.max_steering_rad),
      static_cast<double>(plant.delayed_steering_command_2) + max_steering_step);
  const double throttle_lo = std::max(static_cast<double>(MPC_THROTTLE_MIN_DRIVE),
      static_cast<double>(plant.delayed_throttle_command_2) - max_throttle_step);
  const double throttle_hi = std::min(static_cast<double>(MPC_THROTTLE_MAX),
      static_cast<double>(plant.delayed_throttle_command_2) + max_throttle_step);
  int feasible = 0, evaluated = 0;
  double minimum_slack = INFINITY, maximum_slack = -INFINITY;
  double minimum_ey = INFINITY, maximum_ey = -INFINITY;
  double minimum_epsi = INFINITY, maximum_epsi = -INFINITY;
  int noncorridor_failures = 0;
  constexpr int steering_samples = 17;
  constexpr int throttle_samples = 7;
  for (int si = 0; si < steering_samples; ++si) {
    const double sf = static_cast<double>(si) / (steering_samples - 1);
    const float steering = static_cast<float>(
        steering_lo + sf * (steering_hi - steering_lo));
    for (int ti = 0; ti < throttle_samples; ++ti) {
      const double tf = static_cast<double>(ti) / (throttle_samples - 1);
      const float throttle = static_cast<float>(
          throttle_lo + tf * (throttle_hi - throttle_lo));
      MpcModelControl_t controls[PREDICTION_HORIZON]{};
      std::copy_n(seed_controls, PREDICTION_HORIZON, controls);
      controls[0] = {steering, throttle};
      MpcRtiNominal_t one_step{};
      int failure_stage = -1;
      const MpcRtiRolloutStatus_t status = mpc_rti_validate_candidate_plan(
          &context.initial_state, context.progress_m, controls,
          context.trajectory.data(), context.trajectory.size(),
          context.lap_length_m, 1, context.prediction_dt,
          &validation, &one_step, &failure_stage);
      ++evaluated;
      if (status != MPC_RTI_ROLLOUT_OK &&
          status != MPC_RTI_ROLLOUT_CORRIDOR)
        ++noncorridor_failures;
      if (status == MPC_RTI_ROLLOUT_OK) ++feasible;
      if (failure_stage < 0 && status != MPC_RTI_ROLLOUT_OK) continue;
      MpcRtiReference_t reference{};
      float lower = NAN, upper = NAN;
      if (!mpc_rti_reference_at_progress(context.trajectory.data(),
              context.trajectory.size(), context.lap_length_m,
              one_step.progress[1], &validation.model, &reference) ||
          !mpc_rti_compute_corridor_slack(&one_step.states[1].plant,
              reference.left_bound, reference.right_bound,
              &validation.model, 1, &lower, &upper)) continue;
      const double slack = std::min(lower, upper);
      minimum_slack = std::min(minimum_slack, slack);
      maximum_slack = std::max(maximum_slack, slack);
      minimum_ey = std::min(minimum_ey,
          static_cast<double>(one_step.states[1].plant.e_y));
      maximum_ey = std::max(maximum_ey,
          static_cast<double>(one_step.states[1].plant.e_y));
      minimum_epsi = std::min(minimum_epsi,
          static_cast<double>(one_step.states[1].plant.e_psi));
      maximum_epsi = std::max(maximum_epsi,
          static_cast<double>(one_step.states[1].plant.e_psi));
    }
  }
  std::cout << "{\"recoverability_probe\":\"first_delayed_stage\","
            << "\"actions_evaluated\":" << evaluated
            << ",\"corridor_feasible_actions\":" << feasible
            << ",\"noncorridor_failures\":" << noncorridor_failures
            << ",\"steering_range_rad\":[" << steering_lo << ','
            << steering_hi << "],\"throttle_range\":[" << throttle_lo
            << ',' << throttle_hi << "],\"next_stage_clearance_range_m\":[";
  json_number(minimum_slack); std::cout << ','; json_number(maximum_slack);
  std::cout << "],\"next_stage_e_y_range_m\":[";
  json_number(minimum_ey); std::cout << ','; json_number(maximum_ey);
  std::cout << "],\"next_stage_e_psi_range_rad\":[";
  json_number(minimum_epsi); std::cout << ','; json_number(maximum_epsi);
  std::cout << "],\"initial_actuator_queue\":["
            << plant.delayed_steering_command_1 << ','
            << plant.delayed_steering_command_2 << ','
            << plant.delayed_throttle_command_1 << ','
            << plant.delayed_throttle_command_2 << "]}\n";
}

void emit_pass(const char *name, const MpcRtiQpCapture_t &capture,
               const MpcModelControl_t *nominal_controls,
               const MpcModelControl_t *unscaled_controls,
               const MpcModelControl_t *proposal_controls,
               bool controls_valid,
               const MpcRtiSqpStepMetrics_t &metrics,
               const sdu_apex::qp_fixture::FailureReplayContext &context) {
  RiccatiAdmmState_t state = capture.initial_solver_state;
  RiccatiAdmmConfig_t solver_config = capture.solver_config;
  // The live QP can stop early on a wall-clock deadline. A faster offline
  // machine/thread will run more ADMM steps if that time cap is replayed.
  // Replay the EXACT recorded iteration count to reconstruct the numerical
  // iterate. This reproduces the solver math, not the original clock event.
  const int recorded_iterations = capture.raw_solution.iterations;
  const bool deadline_iteration_replay =
      recorded_iterations > 0 &&
      recorded_iterations < solver_config.max_iterations &&
      capture.raw_status != RICCATI_STATUS_OPTIMAL;
  if (recorded_iterations > 0) {
    /* Even an ADMM run that exhausted its iteration cap can be cut short
     * by a slower replay machine if the recorded wall clock gate is kept.
     * Match the exact mathematical iteration budget for ALL frozen QPs.
     * Only native solver convergence may terminate earlier, and that must
     * then be reported as a parity failure rather than hidden. */
    solver_config.max_iterations = recorded_iterations;
    solver_config.max_solve_time_us = 0.0;
  }
  RiccatiSolution_t replayed{};
  float primal_u[RICCATI_HORIZON][RICCATI_MAX_NU]{};
  float feedback[RICCATI_HORIZON][RICCATI_MAX_NU][RICCATI_MAX_NX]{};
  const RiccatiStatus_t status = riccati_admm_solve_with_feedback(
      capture.problem.steps, capture.problem.terminal_Q,
      capture.problem.terminal_q, capture.problem.terminal_x_lb,
      capture.problem.terminal_x_ub, capture.problem.x0, &solver_config,
      &state, &replayed, primal_u, feedback);
  const double qp_delta = max_solution_delta(replayed, capture.raw_solution);
  const bool qp_exact = status == capture.raw_status &&
      replayed.status == capture.raw_solution.status &&
      replayed.iterations == capture.raw_solution.iterations && qp_delta <= 1.e-6;

  /* Independent nominal hard-footprint QP/source geometry parity: no
   * ADMM approximation or proposed control enters this comparison. */
  {
    auto wide=context.trajectory;
    for(auto &v:wide)v.left_bound=v.right_bound=100.f;
    MpcRtiCycleConfiguration_t validation{};
    validation.model=context.model;
    validation.model.world_sweep_check=nullptr;
    validation.model.world_sweep_context=nullptr;
    MpcRtiNominal_t base{};
    int failed=-1;
    if(mpc_rti_validate_candidate_plan(&context.initial_state,
         context.progress_m,nominal_controls,wide.data(),wide.size(),
         context.lap_length_m,PREDICTION_HORIZON,context.prediction_dt,
         &validation,&base,&failed)==MPC_RTI_ROLLOUT_OK) {
      for(int k:{1,10,20,30}) {
        const auto &plant=base.states[k].plant;
        MpcRtiReference_t ref{};
        const int refok=mpc_rti_reference_at_progress(
             context.trajectory.data(),context.trajectory.size(),
             context.lap_length_m,base.progress[k],&context.model,&ref);
        float lo=NAN,hi=NAN;
        const int ok=refok && mpc_rti_compute_corridor_slack(
             &plant,ref.left_bound,ref.right_bound,&context.model,k,&lo,&hi);
        const auto &poly=capture.problem.state_polygons[k];
        float pmin=INFINITY;
        for(int j=0;j<poly.count;++j) {
          pmin=std::min(pmin,poly.upper[j]-
               poly.normal[j][0]*plant.e_y-poly.normal[j][1]*plant.e_psi);
        }
        std::cout<<"{\"nominal_polygon_parity\":\""<<name
                 <<"\",\"stage\":"<<k<<",\"exact_margin\":"
                 <<(ok?std::min(lo,hi):NAN)
                 <<",\"plane_margin\":"<<pmin
                 <<",\"ey\":"<<plant.e_y<<",\"epsi\":"<<plant.e_psi
                 <<",\"left_bound\":"<<ref.left_bound
                 <<",\"right_bound\":"<<ref.right_bound
                 <<",\"plane_count\":"<<poly.count<<"}\n";
      }
    }
  }
  SourceScore nominal{}, unscaled{}, proposal{};
  MpcRtiSqpStepMetrics_t replay_metrics{};
  bool replay_metrics_valid = false;
  if (controls_valid) {
    nominal = score_source_controls(context, nominal_controls);
    unscaled = score_source_controls(context, unscaled_controls);
    proposal = score_source_controls(context, proposal_controls);
    replay_metrics_valid = mpc_rti_measure_qp_source_agreement(
        &context.initial_state, context.progress_m, nominal_controls,
        proposal_controls, &capture.problem, context.trajectory.data(),
        context.trajectory.size(), context.lap_length_m,
        context.prediction_dt, &context.model, &replay_metrics);
  }
  const auto reduction = [](const SourceScore &base, const SourceScore &trial) {
    return base.complete && trial.complete ? base.objective - trial.objective
                                           : NAN;
  };
  const double unscaled_exact_reduction = reduction(nominal, unscaled);
  const double proposal_exact_reduction = reduction(nominal, proposal);
  const auto &stage_metrics = replay_metrics_valid ? replay_metrics : metrics;
  const double proposal_ratio = metrics.qp_predicted_cost_reduction > 1.e-9
      ? proposal_exact_reduction / metrics.qp_predicted_cost_reduction : NAN;
  constexpr double damping_trials[] = {0.5, 0.25, 0.125, 0.0625};
  SourceScore damped_scores[4]{};
  if (controls_valid) {
    for (int trial = 0; trial < 4; ++trial) {
      MpcModelControl_t damped[PREDICTION_HORIZON]{};
      const double alpha = damping_trials[trial];
      for (int k = 0; k < PREDICTION_HORIZON; ++k) {
        damped[k].steering_target_rad = static_cast<float>(
            nominal_controls[k].steering_target_rad + alpha *
            (proposal_controls[k].steering_target_rad -
             nominal_controls[k].steering_target_rad));
        damped[k].throttle_command = static_cast<float>(
            nominal_controls[k].throttle_command + alpha *
            (proposal_controls[k].throttle_command -
             nominal_controls[k].throttle_command));
      }
      damped_scores[trial] = score_source_controls(context, damped);
    }
  }

  std::cout << "{\"pass\":\"" << name << "\",\"qp_exact_replay\":"
            << (qp_exact ? "true" : "false")
            << ",\"qp_solution_max_abs_delta\":" << qp_delta
            << ",\"qp_status\":" << static_cast<int>(status)
            << ",\"qp_iterations\":" << replayed.iterations
            << ",\"recorded_qp_iterations\":" << capture.raw_solution.iterations
            << ",\"deadline_iteration_replay\":"
            << (deadline_iteration_replay ? "true" : "false")
            << ",\"captured_max_solve_time_us\":"
            << capture.solver_config.max_solve_time_us
            << ",\"proposal_controls_valid\":"
            << (controls_valid ? "true" : "false")
            << ",\"full_horizon_qp_source_comparison_valid\":"
            << (replay_metrics_valid ? "true" : "false")
            << ",\"qp_primal_residual\":";
  json_number(replayed.primal_residual);
  std::cout << ",\"qp_dual_residual\":";
  json_number(replayed.dual_residual);
  std::cout << ",\"qp_predicted_reduction\":";
  json_number(metrics.qp_predicted_cost_reduction);
  std::cout << ",\"qp_unscaled_predicted_reduction\":";
  json_number(metrics.unscaled_qp_predicted_cost_reduction);
  std::cout << ",\"source_exact_reduction\":";
  json_number(proposal_exact_reduction);
  std::cout << ",\"source_exact_unscaled_reduction\":";
  json_number(unscaled_exact_reduction);
  std::cout << ",\"damped_trials\":[";
  for (int trial = 0; trial < 4; ++trial) {
    if (trial) std::cout << ',';
    const auto &damped = damped_scores[trial];
    std::cout << "{\"alpha\":" << damping_trials[trial]
              << ",\"source_status\":\""
              << rollout_name(damped.original_status)
              << "\",\"failure_stage\":"
              << damped.original_failure_stage
              << ",\"source_exact_reduction\":";
    json_number(reduction(nominal, damped));
    std::cout << '}';
  }
  std::cout << ']';
  std::cout << ",\"proposal_agreement_ratio\":";
  json_number(proposal_ratio);
  std::cout << ",\"maximum_state_error\":";
  json_number(stage_metrics.maximum_state_error);
  std::cout << ",\"maximum_state_error_stage\":"
            << stage_metrics.maximum_state_error_stage
            << ",\"maximum_state_error_index\":"
            << stage_metrics.maximum_state_error_index
            << ",\"maximum_footprint_constraint_error_m\":";
  json_number(stage_metrics.maximum_footprint_constraint_error_m);
  std::cout << ",\"maximum_footprint_error_stage\":"
            << stage_metrics.maximum_footprint_error_stage
            << ",\"maximum_source_footprint_violation_m\":";
  json_number(stage_metrics.maximum_source_footprint_violation_m);
  std::cout << ",\"maximum_source_footprint_violation_stage\":"
            << stage_metrics.maximum_source_footprint_violation_stage;
  std::cout << ",\"nominal_rollout_complete\":"
            << (nominal.complete ? "true" : "false")
            << ",\"proposal_rollout_complete\":"
            << (proposal.complete ? "true" : "false")
            << ",\"nominal_objective\":";
  json_number(nominal.objective);
  std::cout << ",\"proposal_objective\":";
  json_number(proposal.objective);
  std::cout << ",\"nominal_original_corridor_status\":\""
            << rollout_name(nominal.original_status)
            << "\",\"nominal_original_corridor_failure_stage\":"
            << nominal.original_failure_stage
            << ",\"original_corridor_status\":\""
            << rollout_name(proposal.original_status)
            << "\",\"original_corridor_failure_stage\":"
            << proposal.original_failure_stage
            << ",\"relaxed_source_status\":\""
            << rollout_name(proposal.relaxed_status)
            << "\",\"relaxed_source_failure_stage\":"
            << proposal.relaxed_failure_stage
            << ",\"source_world_map_callback_replayed\":false"
            << ",\"recorded_world_map_callback_present\":"
            << (context.world_sweep_was_configured ? "true" : "false")
            << ",\"state_error_by_stage\":[";
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    if (k) std::cout << ',';
    json_number(stage_metrics.state_error_by_stage[k]);
  }
  std::cout << "],\"state_error_index_by_stage\":[";
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    if (k) std::cout << ',';
    std::cout << stage_metrics.state_error_index_by_stage[k];
  }
  std::cout << "],\"footprint_constraint_error_by_stage_m\":[";
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    if (k) std::cout << ',';
    json_number(stage_metrics.footprint_constraint_error_by_stage_m[k]);
  }
  std::cout << "],\"source_footprint_violation_by_stage_m\":[";
  for (int k = 0; k <= PREDICTION_HORIZON; ++k) {
    if (k) std::cout << ',';
    json_number(stage_metrics.source_footprint_violation_by_stage_m[k]);
  }
  std::cout << "]}\n";
  if (!qp_exact)
    throw std::runtime_error(std::string(name) + " QP failed exact parity replay");
  if (controls_valid)
    emit_restoration(name, capture, nominal_controls, context);
}

}  // namespace

// Offline-only source-exact Phase-I convergence probe. Never used by the live
// controller and never produces publishable commands without the original
// strict source-corridor validator. Compare both stored SQP initializations.
void emit_source_phase_one(const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcModelControl_t seed[PREDICTION_HORIZON]) {
  MpcRtiCycleConfiguration_t configuration{};
  configuration.model = context.model;
  configuration.model.world_sweep_check = nullptr;
  configuration.model.world_sweep_context = nullptr;
  configuration.model.world_map_rejection_only = 0;
  configuration.objective_schedule = &context.objective;
  MpcRtiSourceNonlinearOptions_t options{};
  // Development-only env sweep; the runtime solver never reads these.
  const auto env_number=[](const char *name,double fallback) {
    const char *p=std::getenv(name);
    return p ? std::strtod(p,nullptr) : fallback;
  };
  options.max_iterations = static_cast<int>(env_number("MPC_PHASE1_OFFLINE_ITER",1));
  options.max_exact_trials_per_iteration = static_cast<int>(
      env_number("MPC_PHASE1_OFFLINE_TRIALS",8));
  options.initial_step_scale = 1.f;
  options.minimum_step_scale = .001f;
  options.armijo_fraction = .0001f;
  options.steering_direction_norm_rad = static_cast<float>(
      env_number("MPC_PHASE1_OFFLINE_STEER_NORM",.20));
  options.throttle_direction_norm = static_cast<float>(
      env_number("MPC_PHASE1_OFFLINE_THROTTLE_NORM",.35));
  options.maximum_wall_time_us = 1000000.; // up to one second, OFFLINE only
  MpcRtiSourceNonlinearResult_t outcome{};
  const int ok = mpc_rti_source_repair_infeasible_nominal(
      &context.initial_state, context.progress_m,
      context.trajectory.data(), context.trajectory.size(),
      context.lap_length_m, context.prediction_dt, &configuration,
      &context.objective, seed, &options, &outcome, nullptr);
  const auto final = ok ? score_source_controls(context, outcome.controls)
                        : SourceScore{};
  std::cout << "{\"source_phase_one_offline_probe\":\"" << name
            << "\",\"valid_full_horizon\":" << (ok ? "true" : "false")
            << ",\"initial_failure_stage\":" << outcome.initial_failure_stage
            << ",\"phase_one_attempted\":" << outcome.feasibility_iterations_attempted
            << ",\"phase_one_accepted\":" << outcome.feasibility_iterations_accepted
            << ",\"initial_corridor_violation_m2\":";
  json_number(outcome.initial_corridor_violation_m2);
  std::cout << ",\"final_corridor_violation_m2\":";
  json_number(outcome.final_corridor_violation_m2);
  std::cout << ",\"iterations_accepted\":" << outcome.iterations_accepted
            << ",\"source_evaluations\":" << outcome.exact_rollout_evaluations
            << ",\"wall_ms\":" << outcome.wall_time_ms
            << ",\"strict_status\":\"" << rollout_name(final.original_status)
            << "\",\"strict_minimum_clearance_m\":";
  json_number(final.minimum_clearance);
  std::cout << "}\n";
}

// Offline-only experiment: a single source-adjoint racing-cost SQP
// globalization from an accepted alpha=0 nominal, independently validated
// against the ORIGINAL strict source footprint. No live controller output.
void emit_source_gradient_probe(const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcModelControl_t nominal[PREDICTION_HORIZON]) {
  MpcRtiCycleConfiguration_t configuration{};
  configuration.model=context.model;
  configuration.model.world_sweep_check=nullptr;
  configuration.model.world_sweep_context=nullptr;
  configuration.model.world_map_rejection_only=0;
  configuration.objective_schedule=&context.objective;
  MpcRtiSourceNonlinearOptions_t options{};
  options.max_iterations=1;
  const char *trials=std::getenv("MPC_SOURCE_GRADIENT_TRIALS");
  options.max_exact_trials_per_iteration=trials
      ? std::clamp(std::atoi(trials),1,16):4;
  const char *initial_scale=std::getenv("MPC_SOURCE_GRADIENT_INITIAL_SCALE");
  options.initial_step_scale=initial_scale
      ? static_cast<float>(std::atof(initial_scale)):1.f;
  options.minimum_step_scale=.0001f;
  options.armijo_fraction=.0001f;
  options.steering_direction_norm_rad=.15f;
  options.throttle_direction_norm=.25f;
  options.directional_check_step_scale=0.f;
  options.maximum_wall_time_us=1000000.; // offline only
  MpcRtiSourceNonlinearResult_t result{};
  const bool ok=mpc_rti_source_nonlinear_optimize(
      &context.initial_state,context.progress_m,
      context.trajectory.data(),context.trajectory.size(),
      context.lap_length_m,context.prediction_dt,
      &configuration,&context.objective,nominal,&options,&result,nullptr);
  const auto certified=ok && result.final_valid &&
      result.iterations_accepted>0
      ? score_source_controls(context,result.controls):SourceScore{};
  std::cout<<"{\"source_gradient_offline_probe\":\""<<name
           <<"\",\"nominal_strict_valid\":"
           <<(result.initial_valid?"true":"false")
           <<",\"gradient_optimizer_valid\":"<<(ok?"true":"false")
           <<",\"iterations_accepted\":"<<result.iterations_accepted
           <<",\"source_evaluations\":"<<result.exact_rollout_evaluations
           <<",\"adjoint_evaluations\":"<<result.source_adjoint_evaluations
           <<",\"wall_ms\":"<<result.wall_time_ms
           <<",\"initial_objective\":";
  json_number(result.initial_objective);
  std::cout<<",\"final_objective\":";
  json_number(result.final_objective);
  std::cout<<",\"minimum_clearance_m\":";
  json_number(result.minimum_corridor_clearance_m);
  std::cout<<",\"certified_strict_status\":\""
           <<rollout_name(certified.original_status)
           <<"\",\"steering_first_delta_rad\":";
  json_number(result.iterations_accepted>0
      ?result.controls[0].steering_target_rad-nominal[0].steering_target_rad
      :NAN);
  std::cout<<",\"throttle_first_delta\":";
  json_number(result.iterations_accepted>0
      ?result.controls[0].throttle_command-nominal[0].throttle_command
      :NAN);
  std::cout<<"}\n";
}

// Offline-only source-certified fast descent from the *already formed* QP
// full-horizon gradient; no second source LTV build or adjoint is required.
// This measures whether the source-gradient optimization can be generated
// from the original R1/R2 QP with substantially less arithmetic.
void emit_qp_gradient_probe(const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcRtiQpCapture_t &capture,
    const MpcModelControl_t nominal[PREDICTION_HORIZON]) {
  double grad[PREDICTION_HORIZON][MPC_RTI_NU]{};
  const int computed=mpc_rti_qp_source_gradient(&capture.problem,nominal,
      context.model.command_increment_qp_enabled,grad);
  double norm[MPC_RTI_NU]{};
  if(computed) {
    for(int k=0;k<PREDICTION_HORIZON;++k)
      for(int a=0;a<MPC_RTI_NU;++a)
        norm[a]+=grad[k][a]*grad[k][a];
    for(auto &n:norm)n=std::sqrt(n);
  }
  const auto baseline=score_source_controls(context,nominal);
  constexpr double steps[]={.0625,.03125};
  int strict_feasible=0,improvements=0;
  double best=baseline.objective,best_scale=NAN, first_delta=NAN;
  for(double t:steps) {
    MpcModelControl_t trial[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
      trial[k].steering_target_rad=std::clamp(
          static_cast<float>(nominal[k].steering_target_rad-
              t*.15*grad[k][0]/std::max(norm[0],1.e-8)),
          -context.model.max_steering_rad,context.model.max_steering_rad);
      trial[k].throttle_command=std::clamp(
          static_cast<float>(nominal[k].throttle_command-
              t*.25*grad[k][1]/std::max(norm[1],1.e-8)),
          MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
    }
    const auto exact=score_source_controls(context,trial);
    if(exact.original_status==MPC_RTI_ROLLOUT_OK) {
      ++strict_feasible;
      if(exact.complete && exact.objective<best-1.e-5) {
        best=exact.objective;
        best_scale=t;
        first_delta=trial[0].steering_target_rad-nominal[0].steering_target_rad;
        ++improvements;
      }
    }
  }
  std::cout<<"{\"qp_gradient_offline_probe\":\""<<name
           <<"\",\"computed\":"<<(computed?"true":"false")
           <<",\"nominal_strict_status\":\""
           <<rollout_name(baseline.original_status)
           <<"\",\"strict_feasible_trials\":"<<strict_feasible
           <<",\"improvements\":"<<improvements
           <<",\"best_exact_source_cost\":";
  json_number(best);
  std::cout<<",\"nominal_cost\":";
  json_number(baseline.objective);
  std::cout<<",\"best_scale\":";
  json_number(best_scale);
  std::cout<<",\"first_steering_delta\":";
  json_number(first_delta);
  std::cout<<"}\n";
}

// Offline-only local source/QP corridor-slack adjoint. Derive a physical
// clearance-improving direction from the closest active footprint half-plane,
// reuse the captured source LTV A/B, and certify every proposed 750ms plan
// with the original nonlinear geometry and full delayed command queue.
// Research diagnostic only: no control publication and no relaxed walls.
void emit_clearance_gradient_probe(const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcRtiQpCapture_t &qp,
    const MpcModelControl_t controls[PREDICTION_HORIZON]) {
  MpcRtiCycleConfiguration_t config{};
  config.model=context.model;
  config.model.world_sweep_check=nullptr;
  config.model.world_sweep_context=nullptr;
  config.objective_schedule=&context.objective;
  auto nominal=std::make_unique<MpcRtiNominal_t>();
  int failure=-1;
  const auto validated=mpc_rti_validate_candidate_plan(
      &context.initial_state,context.progress_m,controls,
      context.trajectory.data(),context.trajectory.size(),
      context.lap_length_m,PREDICTION_HORIZON,context.prediction_dt,
      &config,nominal.get(),&failure);
  if(validated!=MPC_RTI_ROLLOUT_OK) {
    std::cout<<"{\"clearance_gradient_offline_probe\":\""<<name
             <<"\",\"nominal_strict_valid\":false}\n";
    return;
  }
  double gradient[PREDICTION_HORIZON][MPC_RTI_NU]{};
  float closest=INFINITY;
  int closest_stage=-1;
  if(!mpc_rti_qp_clearance_gradient(&qp.problem,nominal.get(),
        context.model.command_increment_qp_enabled,gradient,&closest,
        &closest_stage)) {
    std::cout<<"{\"clearance_gradient_offline_probe\":\""<<name
             <<"\",\"gradient_available\":false}\n";
    return;
  }
  double magnitude[2]{};
  for(int k=0;k<PREDICTION_HORIZON;++k)
    for(int j=0;j<2;++j)
      magnitude[j]+=gradient[k][j]*gradient[k][j];
  magnitude[0]=std::sqrt(magnitude[0]);
  magnitude[1]=std::sqrt(magnitude[1]);
  constexpr double scales[]={.03125,.0625,.125,.25,.5,1.,2.,4.};
  double best_slack=nominal->validation_minimum_corridor_slack_m;
  double best_objective=nominal->validation_objective;
  double best_scale=NAN;
  int legal_trials=0,improved_slack=0,improved_cost_with_reserve=0;
  for(double scale:scales) {
    MpcModelControl_t trial[PREDICTION_HORIZON]{};
    for(int k=0;k<PREDICTION_HORIZON;++k) {
      trial[k].steering_target_rad=std::clamp(
          float(controls[k].steering_target_rad+scale*.15*
              gradient[k][0]/std::max(magnitude[0],1.e-8)),
          -context.model.max_steering_rad,context.model.max_steering_rad);
      trial[k].throttle_command=std::clamp(
          float(controls[k].throttle_command+scale*.25*
              gradient[k][1]/std::max(magnitude[1],1.e-8)),
          MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
    }
    auto candidate=std::make_unique<MpcRtiNominal_t>();
    int failed=-1;
    if(mpc_rti_validate_candidate_plan(&context.initial_state,
        context.progress_m,trial,context.trajectory.data(),
        context.trajectory.size(),context.lap_length_m,
        PREDICTION_HORIZON,context.prediction_dt,&config,
        candidate.get(),&failed)!=MPC_RTI_ROLLOUT_OK)continue;
    ++legal_trials;
    if(candidate->validation_minimum_corridor_slack_m>
        nominal->validation_minimum_corridor_slack_m+.001f)++improved_slack;
    if(candidate->validation_minimum_corridor_slack_m>best_slack &&
       candidate->validation_objective<=1.02*nominal->validation_objective) {
      best_slack=candidate->validation_minimum_corridor_slack_m;
      best_objective=candidate->validation_objective;
      best_scale=scale;
      ++improved_cost_with_reserve;
    }
  }
  std::cout<<"{\"clearance_gradient_offline_probe\":\""<<name
           <<"\",\"nominal_strict_valid\":true"
           <<",\"limiting_stage\":"<<closest_stage
           <<",\"linearized_closest_slack_m\":"<<closest
           <<",\"nominal_exact_min_slack_m\":"
           <<nominal->validation_minimum_corridor_slack_m
           <<",\"nominal_source_cost\":"
           <<nominal->validation_objective
           <<",\"legal_trials\":"<<legal_trials
           <<",\"clearance_improved_trials\":"<<improved_slack
           <<",\"accepted_clearance_updates\":"<<improved_cost_with_reserve
           <<",\"best_clearance_under_2pct_cost\":"<<best_slack
           <<",\"best_cost_under_2pct\":"<<best_objective
           <<",\"best_scale\":";
  json_number(best_scale);
  std::cout<<"}\n";
}

/* Directional first-order and finite-step QP/source agreement on the EXACT
 * frozen snapshot: independent of alpha acceptance or a warm-start reset.
 * The live optimization must produce descent directions, not merely steps
 * that look good in its own mis-linearized QP model. */
// Separate incorrect first-order plant sensitivities from objective-term
// mismatches: rebuild the frozen QP with one cost family disabled and score
// the same exact-source directional finite differences with that family off.
void emit_sqp_cost_attribution(const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcModelControl_t nominal[PREDICTION_HORIZON],
    const MpcModelControl_t proposal[PREDICTION_HORIZON]) {
  auto built=std::make_unique<MpcRtiNominal_t>();
  MpcRtiReference_t refs[PREDICTION_HORIZON+1]{};
  auto base=context;
  base.model.world_sweep_check=nullptr;
  base.model.world_sweep_context=nullptr;
  if(!mpc_rti_build_nominal_same_epoch(&base.initial_state,base.progress_m,
      nullptr,nominal,PREDICTION_HORIZON,base.trajectory.data(),
      base.trajectory.size(),base.lap_length_m,base.prediction_dt,
      PREDICTION_HORIZON,&base.model,built.get(),refs)) {
    std::cout<<"{\"sqp_cost_attribution\":\""<<name
             <<"\",\"rebuild_nominal\":false}\n";return;
  }
  const double eps=.001;
  MpcModelControl_t plus[PREDICTION_HORIZON]{},minus[PREDICTION_HORIZON]{};
  for(int k=0;k<PREDICTION_HORIZON;++k) {
    const double ds=proposal[k].steering_target_rad-nominal[k].steering_target_rad;
    const double dt=proposal[k].throttle_command-nominal[k].throttle_command;
    plus[k]={float(nominal[k].steering_target_rad+eps*ds),
             float(nominal[k].throttle_command+eps*dt)};
    minus[k]={float(nominal[k].steering_target_rad-eps*ds),
              float(nominal[k].throttle_command-eps*dt)};
  }
  const char* groups[]={"all","lateral","heading","speed","slip",
    "lateral_speed","yaw","steering_command","steering_target",
    "throttle_command","steering_slew","throttle_slew","corridor_reserve"};
  std::cout<<"{\"sqp_cost_attribution\":\""<<name<<"\",\"terms\":[";
  for(int g=0;g<13;++g) {
    auto c=base;
    auto &m=c.model;
    if(g==1)m.weight_e_y=0;
    if(g==2)m.weight_e_psi=0;
    if(g==3) {
      m.weight_u=m.weight_u_overspeed=0;
      for(auto &x:c.objective.speed_weights)x=0.;
    }
    if(g==4)m.weight_longitudinal_slip=0;
    if(g==5)m.weight_v=0;
    if(g==6)m.weight_r=0;
    if(g==7)m.weight_steering_command=0;
    if(g==8)m.weight_steering_target=0;
    if(g==9)m.weight_throttle_command=0;
    if(g==10)m.weight_steering_slew=m.weight_steering_correction_variation=0;
    if(g==11)m.weight_throttle_slew=0;
    if(g==12)m.weight_source_corridor_reserve=0;
    const auto qp=std::make_unique<MpcRtiProblem_t>();
    const int ok=mpc_rti_build_ltv_qp_with_objective_schedule(
        built->states,built->controls,refs,PREDICTION_HORIZON,
        c.prediction_dt,&m,&c.objective,qp.get());
    double gradient[PREDICTION_HORIZON][MPC_RTI_NU]{};
    double qp_deriv=NAN;
    if(ok && mpc_rti_qp_source_gradient(qp.get(),nominal,
        m.command_increment_qp_enabled,gradient)) {
      qp_deriv=0.;
      for(int k=0;k<PREDICTION_HORIZON;++k) {
        qp_deriv+=gradient[k][0]*(proposal[k].steering_target_rad-nominal[k].steering_target_rad);
        qp_deriv+=gradient[k][1]*(proposal[k].throttle_command-nominal[k].throttle_command);
      }
    }
    const auto sp=score_source_controls(c,plus);
    const auto sm=score_source_controls(c,minus);
    const double exact=sp.complete&&sm.complete?
        (sp.objective-sm.objective)/(2.*eps):NAN;
    if(g)std::cout<<',';
    std::cout<<"{\"name\":\""<<groups[g]<<"\",\"qp\":";
    json_number(qp_deriv);
    std::cout<<",\"source\":";json_number(exact);
    std::cout<<"}";
  }
  std::cout<<"]}\n";
}

void emit_sqp_directional_consistency(const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcRtiQpCapture_t &capture,
    const MpcModelControl_t nominal[PREDICTION_HORIZON],
    const MpcModelControl_t proposal[PREDICTION_HORIZON]) {
    double gradient[PREDICTION_HORIZON][MPC_RTI_NU]{};
    if(!mpc_rti_qp_source_gradient(&capture.problem,nominal,
        context.model.command_increment_qp_enabled,gradient))return;
    const auto baseline=score_source_controls(context,nominal);
    double gdot=0.,norm2=0.,max_steer=0.,max_throttle=0.;
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        const double dx=proposal[k].steering_target_rad-nominal[k].steering_target_rad;
        const double dt=proposal[k].throttle_command-nominal[k].throttle_command;
        gdot+=gradient[k][0]*dx+gradient[k][1]*dt;
        norm2+=dx*dx+dt*dt;
        max_steer=std::max(max_steer,std::abs(dx));
        max_throttle=std::max(max_throttle,std::abs(dt));
    }
    std::cout<<"{\"sqp_directional_consistency\":\""<<name<<"\""
             <<",\"baseline_complete\":"<<(baseline.complete?"true":"false")
             <<",\"baseline_hard_valid\":"
             <<(baseline.original_status==MPC_RTI_ROLLOUT_OK?"true":"false")
             <<",\"baseline_exact_cost\":";
    json_number(baseline.objective);
    std::cout<<",\"qp_directional_gradient\":";json_number(gdot);
    std::cout<<",\"direction_l2\":"<<std::sqrt(norm2)
             <<",\"max_steering_step_rad\":"<<max_steer
             <<",\"max_throttle_step\":"<<max_throttle
             <<",\"scales\":[";
    const double scales[]={-.02,-.01,-.005,-.001,0.,.001,.005,.01,.02,.05,.125,.25,.5,1.};
    for(int j=0;j<14;++j) {
        const double a=scales[j];
        MpcModelControl_t trial[PREDICTION_HORIZON]{};
        for(int k=0;k<PREDICTION_HORIZON;++k) {
            trial[k].steering_target_rad=float(nominal[k].steering_target_rad+
              a*(proposal[k].steering_target_rad-nominal[k].steering_target_rad));
            trial[k].throttle_command=float(nominal[k].throttle_command+
              a*(proposal[k].throttle_command-nominal[k].throttle_command));
        }
        const auto source=score_source_controls(context,trial);
        if(j)std::cout<<',';
        std::cout<<"{\"alpha\":"<<a<<",\"complete\":"
                 <<(source.complete?"true":"false")
                 <<",\"hard_valid\":"
                 <<(source.original_status==MPC_RTI_ROLLOUT_OK?"true":"false")
                 <<",\"hard_failure_stage\":"<<source.original_failure_stage
                 <<",\"cost_delta\":";
        json_number(source.complete&&baseline.complete?
            source.objective-baseline.objective:NAN);
        std::cout<<"}";
    }
    std::cout<<"]}\n";
}


/* Actual 30-stage SQP direction: compare the native QP's directional
 * first derivative, the actual exact-source first derivative and the
 * source-derived finite-difference second derivative of the SAME frozen
 * nonlinear racing objective. Crucially, every trial is a complete
 * physically integrated source horizon; an invalid hard-corridor trial
 * is scored only for diagnostics and can never be published.
 *
 * A second-order Taylor correction is useful only if it improves the
 * FINITE correction's predicted cost without introducing adverse outliers.
 * A local curvature result alone does not prove the SQP is globally valid. */
void emit_sqp_directional_second_order(
    const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcRtiQpCapture_t &capture,
    const MpcModelControl_t nominal[PREDICTION_HORIZON],
    const MpcModelControl_t proposal[PREDICTION_HORIZON])
{
    double gradient[PREDICTION_HORIZON][MPC_RTI_NU]{};
    if(!mpc_rti_qp_source_gradient(&capture.problem,nominal,
        context.model.command_increment_qp_enabled,gradient))return;
    double qp_gdot=0.;
    for(int k=0;k<PREDICTION_HORIZON;++k) {
        qp_gdot+=gradient[k][0]*(proposal[k].steering_target_rad-
            nominal[k].steering_target_rad);
        qp_gdot+=gradient[k][1]*(proposal[k].throttle_command-
            nominal[k].throttle_command);
    }
    auto evaluate=[&](double alpha) {
        MpcModelControl_t controls[PREDICTION_HORIZON]{};
        for(int k=0;k<PREDICTION_HORIZON;++k) {
            const double ds=proposal[k].steering_target_rad-
                            nominal[k].steering_target_rad;
            const double dt=proposal[k].throttle_command-
                            nominal[k].throttle_command;
            controls[k].steering_target_rad=
                float(nominal[k].steering_target_rad+alpha*ds);
            controls[k].throttle_command=
                float(nominal[k].throttle_command+alpha*dt);
        }
        return score_source_controls(context,controls);
    };
    const auto baseline=evaluate(0.);
    if(!baseline.complete)return;
    const double h_values[]={.001,.002,.005,.01,.02,.04};
    double best_h=NAN,best_source_g=NAN,best_source_curvature=NAN;
    // Choose the largest tested scale with a converging directional
    // derivative and curvature estimate. This is a diagnostic selection
    // using neighboring estimates, not oracle knowledge of future alpha.
    double previous_g=NAN,previous_hessian=NAN;
    std::cout<<"{\"sqp_directional_second_order\":\""<<name
             <<"\",\"baseline_cost\":";json_number(baseline.objective);
    std::cout<<",\"qp_gradient\":";json_number(qp_gdot);
    std::cout<<",\"curvature_by_step\":[";
    for(int i=0;i<6;++i) {
        const double h=h_values[i];
        const auto plus=evaluate(h),minus=evaluate(-h);
        const bool ok=plus.complete&&minus.complete;
        const double actual_gradient=ok?
            (plus.objective-minus.objective)/(2.*h):NAN;
        const double curvature=ok?
            (plus.objective-2.*baseline.objective+minus.objective)/(h*h):NAN;
        const double scale=std::max(1.,std::abs(actual_gradient));
        if(ok && std::isfinite(previous_g) &&
           std::abs(actual_gradient-previous_g)<.1*scale &&
           std::abs(curvature-previous_hessian)<
               .3*std::max(1.,std::abs(curvature))) {
            best_h=h;best_source_g=actual_gradient;
            best_source_curvature=curvature;
        }
        previous_g=actual_gradient;
        previous_hessian=curvature;
        if(i)std::cout<<',';
        std::cout<<"{\"h\":"<<h<<",\"valid\":"
                 <<(ok?"true":"false")<<",\"source_gradient\":";
        json_number(actual_gradient);
        std::cout<<",\"source_hessian_direction\":";
        json_number(curvature);
        std::cout<<"}";
    }
    std::cout<<"],\"stable_h\":";json_number(best_h);
    std::cout<<",\"stable_source_gradient\":";json_number(best_source_g);
    std::cout<<",\"stable_source_curvature\":";
    json_number(best_source_curvature);
    std::cout<<",\"steps\":[";
    const double scales[]={.03125,.0625,.125,.25,.5,1.};
    for(int i=0;i<6;++i) {
        const double alpha=scales[i];
        const auto trial=evaluate(alpha);
        const double actual=trial.complete?
            trial.objective-baseline.objective:NAN;
        const double linear=qp_gdot*alpha;
        const double quadratic=std::isfinite(best_source_curvature)?
            linear+.5*best_source_curvature*alpha*alpha:NAN;
        if(i)std::cout<<',';
        std::cout<<"{\"alpha\":"<<alpha
                 <<",\"complete\":"<<(trial.complete?"true":"false")
                 <<",\"hard_valid\":"
                 <<(trial.original_status==MPC_RTI_ROLLOUT_OK?"true":"false")
                 <<",\"hard_failure_stage\":"<<trial.original_failure_stage
                 <<",\"actual_cost_delta\":";
        json_number(actual);
        std::cout<<",\"qp_first_order_delta\":";json_number(linear);
        std::cout<<",\"source_second_order_delta\":";
        json_number(quadratic);
        std::cout<<",\"first_error\":";json_number(
            std::isfinite(actual)?std::abs(linear-actual):NAN);
        std::cout<<",\"second_error\":";json_number(
            std::isfinite(actual)&&std::isfinite(quadratic)?
                std::abs(quadratic-actual):NAN);
        std::cout<<"}";
    }
    std::cout<<"]}\n";
}

void emit_source_successor_probe(
    const char *name,
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcModelControl_t *controls)
{
    MpcRtiCycleConfiguration_t cfg{};
    cfg.model=context.model;
    cfg.model.world_sweep_check=nullptr;
    cfg.model.world_sweep_context=nullptr;
    cfg.objective_schedule=nullptr;
    auto nominal=std::make_unique<MpcRtiNominal_t>();
    int failure=-1;
    const auto status=mpc_rti_validate_candidate_plan(
        &context.initial_state,context.progress_m,controls,
        context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,PREDICTION_HORIZON,context.prediction_dt,
        &cfg,nominal.get(),&failure);
    if(status!=MPC_RTI_ROLLOUT_OK) {
        std::cout<<"{\"successor_probe\":\""<<name
                 <<"\",\"initial_full_horizon_valid\":false}\n";
        return;
    }
    const auto before=std::chrono::steady_clock::now();
    MpcRtiSuccessorProbe_t result{};
    const int ok=mpc_rti_source_successor_probe(
        &nominal->states[PREDICTION_HORIZON],
        nominal->progress[PREDICTION_HORIZON],
        nominal->controls[PREDICTION_HORIZON-1],
        context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,context.prediction_dt,&cfg,&result,
        std::getenv("MPC_SUCCESSOR_OFFLINE_STEPS") ?
            std::clamp(std::atoi(std::getenv("MPC_SUCCESSOR_OFFLINE_STEPS")),8,16)
            : 8,
        std::getenv("MPC_SUCCESSOR_OFFLINE_CANDIDATES") &&
        std::atoi(std::getenv("MPC_SUCCESSOR_OFFLINE_CANDIDATES"))==10?10:
        std::getenv("MPC_SUCCESSOR_OFFLINE_CANDIDATES") &&
        std::atoi(std::getenv("MPC_SUCCESSOR_OFFLINE_CANDIDATES"))==3?3:9);
    const double elapsed=std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-before).count();
    const int held=mpc_rti_terminal_extension_probe(
        nominal.get(),context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,context.prediction_dt,&cfg);
    std::cout<<"{\"successor_probe\":\""<<name
             <<"\",\"initial_full_horizon_valid\":true"
             <<",\"probe_computed\":"<<(ok?"true":"false")
             <<",\"optimized_200ms_feasible\":"
             <<(result.feasible?"true":"false")
             <<",\"successor_steps\":"<<result.horizon
             <<",\"candidate_count\":"<<result.candidates_evaluated
             <<",\"feasible_count\":"<<result.feasible_candidates
             <<",\"baseline_held_75ms_probe\":"<<(held?"true":"false")
             <<",\"probe_wall_us\":";
    json_number(elapsed);
    std::cout<<",\"best_clearance_m\":";
    json_number(result.feasible?result.minimum_clearance_m:NAN);
    std::cout<<",\"best_progress_m\":";
    json_number(result.feasible?result.progress_m:NAN);
    std::cout<<",\"terminal_heading_error_rad\":";
    json_number(result.feasible?result.terminal_heading_error_rad:NAN);
    std::cout<<"}\n";
}

void emit_frozen_progress_geometry(
    const char *name,const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcModelControl_t nominal_commands[PREDICTION_HORIZON]) {
  auto wide=context.trajectory;
  for(auto &sample:wide)sample.left_bound=sample.right_bound=100.f;
  MpcRtiCycleConfiguration_t validation{};
  validation.model=context.model;
  validation.model.world_sweep_check=nullptr;
  validation.model.world_sweep_context=nullptr;
  MpcRtiNominal_t nominal{};
  int failed=-1;
  if(mpc_rti_validate_candidate_plan(&context.initial_state,
      context.progress_m,nominal_commands,wide.data(),wide.size(),
      context.lap_length_m,PREDICTION_HORIZON,context.prediction_dt,
      &validation,&nominal,&failed)!=MPC_RTI_ROLLOUT_OK)return;
  constexpr double h=.002;
  for(int k=1;k<=PREDICTION_HORIZON;++k) {
    MpcRtiReference_t plus{},minus{};
    if(!mpc_rti_reference_at_progress(context.trajectory.data(),
        context.trajectory.size(),context.lap_length_m,
        nominal.progress[k]+h,&context.model,&plus) ||
       !mpc_rti_reference_at_progress(context.trajectory.data(),
        context.trajectory.size(),context.lap_length_m,
        nominal.progress[k]-h,&context.model,&minus))continue;
    std::cout<<"{\"frozen_progress_geometry\":\""<<name
        <<"\",\"stage\":"<<k
        <<",\"progress_m\":"<<nominal.progress[k]
        <<",\"left_width_slope\":"
        <<(plus.left_bound-minus.left_bound)/(2*h)
        <<",\"right_width_slope\":"
        <<(plus.right_bound-minus.right_bound)/(2*h)<<"}\n";
  }
}


/* Compare an infinitesimal wall-width derivative to the ACTUAL finite
 * progress difference between two source-integrated SQP nominals. This is
 * the essential check for the preview-minimum nonsmooth corridor function. */
void emit_progress_secant_audit(
    const sdu_apex::qp_fixture::FailureReplayContext &context) {
  if(!context.r1_captured || !context.r2_captured) return;
  auto wide=context.trajectory;
  for(auto &sample:wide)sample.left_bound=sample.right_bound=100.f;
  MpcRtiCycleConfiguration_t c{};
  c.model=context.model;
  c.model.world_sweep_check=nullptr;
  c.model.world_sweep_context=nullptr;
  auto first=std::make_unique<MpcRtiNominal_t>();
  auto second=std::make_unique<MpcRtiNominal_t>();
  int failed=-1;
  if(mpc_rti_validate_candidate_plan(&context.initial_state,context.progress_m,
      context.r1_nominal,wide.data(),wide.size(),context.lap_length_m,
      PREDICTION_HORIZON,context.prediction_dt,&c,first.get(),&failed)!=MPC_RTI_ROLLOUT_OK ||
     mpc_rti_validate_candidate_plan(&context.initial_state,context.progress_m,
      context.r2_nominal,wide.data(),wide.size(),context.lap_length_m,
      PREDICTION_HORIZON,context.prediction_dt,&c,second.get(),&failed)!=MPC_RTI_ROLLOUT_OK)return;
  double max_width_linearization=0.,max_s=0.;
  int max_stage=-1,nontrivial=0;
  for(int k=1;k<=PREDICTION_HORIZON;++k) {
    const double s0=first->progress[k],s1=second->progress[k];
    const double delta=s0-s1;
    MpcRtiReference_t old{},now{},plus{},minus{};
    constexpr double eps=.002;
    if(!mpc_rti_reference_at_progress(context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,s0,&context.model,&old) ||
       !mpc_rti_reference_at_progress(context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,s1,&context.model,&now) ||
       !mpc_rti_reference_at_progress(context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,s1+eps,&context.model,&plus) ||
       !mpc_rti_reference_at_progress(context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,s1-eps,&context.model,&minus))continue;
    const double ll=(plus.left_bound-minus.left_bound)/(2*eps);
    const double lr=(plus.right_bound-minus.right_bound)/(2*eps);
    const double discrepancy=std::max(std::abs(now.left_bound+ll*delta-old.left_bound),
                                     std::abs(now.right_bound+lr*delta-old.right_bound));
    max_s=std::max(max_s,std::abs(delta));
    if(std::abs(delta)>.01)++nontrivial;
    if(discrepancy>max_width_linearization){
      max_width_linearization=discrepancy;max_stage=k;
    }
    if(discrepancy>.01) {
      std::cout<<"{\"progress_width_finite_step\":true,\"stage\":"<<k
       <<",\"source_station_delta\":"<<delta
       <<",\"left_width_secant\":";
      json_number(std::abs(delta)>1.e-8?(old.left_bound-now.left_bound)/delta:NAN);
      std::cout<<",\"left_width_local_slope\":"<<ll
       <<",\"right_width_secant\":";
      json_number(std::abs(delta)>1.e-8?(old.right_bound-now.right_bound)/delta:NAN);
      std::cout<<",\"right_width_local_slope\":"<<lr
       <<",\"max_width_error\":"<<discrepancy<<"}\n";
    }
  }
  std::cout<<"{\"progress_width_secant_summary\":true"
       <<",\"max_linearization_error_m\":"<<max_width_linearization
       <<",\"max_stage\":"<<max_stage
       <<",\"max_progress_delta_m\":"<<max_s
       <<",\"nontrivial_progress_changes\":"<<nontrivial<<"}\n";
}

void emit_normal_sqp_comparison(
    const sdu_apex::qp_fixture::FailureReplayContext &context,
    const MpcRtiQpCapture_t &r1,int selected_repair=-1) {
  for (int repair=0;repair<=27;++repair) {
    if(selected_repair>=0 && repair!=selected_repair)continue;
    // These retired curvature-only variants were unqualified and have
    // been removed from the live optimizer. Preserve other experiment IDs.
    if (repair==3 || repair==5 || repair==7 || repair==9 ||
        repair==10 || repair==12)continue;
    if(const char *selected=selected_repair>=0
         ?nullptr:std::getenv("MPC_REPLAY_SELECTED_MODES")) {
      const std::string patterns=std::string(",")+selected+",";
      if(patterns.find(","+std::to_string(repair)+",")==std::string::npos)
        continue;
    }
    MpcRtiCycleConfiguration_t config{};
    config.model = context.model;
    config.model.world_sweep_check = nullptr;
    config.model.world_sweep_context = nullptr;
    config.model.world_map_rejection_only = 0;
    config.solver = r1.solver_config;
    config.solver.max_solve_time_us = 0.0;
    config.solver.reuse_same_epoch = 1;
    config.max_sqp_iterations=(repair>=4?2:1);
    config.source_violation_filter_sqp=repair>=11;
    if(repair==11)config.source_elastic_wall_weight=8000.f;
    config.progress_coupled_hard_corridor=repair!=6 && repair!=7;
    if(repair>=13) {
      config.source_elastic_wall_weight=
          repair==26?0.f:repair==13?100.f:repair==14?500.f:repair==15?1000.f:
          repair==16?4000.f:repair==20?1000.f:
          (repair==21 || repair==24)?1000.f:
          (repair==22 || repair==25)?500.f:8000.f;
      if(repair==17 || repair==20) {
        config.solver.rho_max=127.f;
        for(int axis:{MPC_OPT_IDX_EY,MPC_OPT_IDX_EPSI,
                      MPC_OPT_IDX_PROGRESS_DEVIATION})
          config.solver.rho_x_by_state[axis]=32.f;
      }
      if(repair==18 || repair==20)
        config.solver.max_iterations=300;
      if(repair==19)config.solver.adaptive_rho=1;
      if(repair>=21 && repair<=25) {
        config.solver.rho=127.f;
        config.solver.rho_u=127.f;
        config.solver.rho_max=127.f;
        config.solver.adaptive_rho=0;
        config.solver.max_iterations=repair==23?600:300;
        config.solver.tolerance=repair>=24?.0005f:.002f;
        for(auto &entry:config.solver.rho_x_by_state)entry=127.f;
        for(auto &entry:config.solver.rho_u_by_input)entry=127.f;
      }
    }
    if(repair==27) {
      // Native 60-control SQP entered through the EXACT controller entry,
      // not the standalone reduced-SQP offline replay driver. Old R1/R2
      // QP, elastic restoration and legacy gradients must not execute.
      config.native_source_reduced_sqp=1;
      config.native_source_reduced_sqp_iterations=8;
      config.source_violation_filter_sqp=0;
      config.progress_coupled_hard_corridor=0;
    }
    if(repair>=8 && repair!=27) {
      config.model.command_increment_qp_enabled=1;
      config.model.source_reachable_steering_qp=1;
      config.model.max_steering_command_step_rad=2.f*SOURCE_MAX_STEERING_RAD;
      config.model.max_throttle_command_step=1.f;
    }

    config.command_epoch_ns=context.command_epoch_ns;
    config.line_search_command_epoch_valid=1;
    config.enable_nonlinear_line_search=1;
    config.line_search_max_backtracks=4;
    config.feedback_rollout=0;
    config.feedback_rollout_gain_scale=1.f;
    config.source_nominal_feasibility_repair=repair==1 || repair==2;
    if (repair==2) {
      config.enable_feasibility_restoration=1;
      config.prefer_feasibility_first=1;
      config.feasibility_restoration_qp_relaxation_m=.18f;
      config.feasibility_restoration_wall_weight=8000.f;
    }
    auto memory=std::make_unique<MpcRtiMemory_t>();
    auto result=std::make_unique<MpcRtiCycleResult_t>();
    auto exported_first=std::make_unique<MpcRtiQpCapture_t>();
    auto exported_second=std::make_unique<MpcRtiQpCapture_t>();
    if(std::getenv("MPC_FROZEN_ELASTIC_CAPTURE_DIR") && repair>=11) {
      config.qp_capture=exported_first.get();
      config.qp_capture_second_pass=exported_second.get();
    }
    memory->command_epoch_ns=context.command_epoch_ns;
    memory->warm_start_controls_valid=1;
    memory->warm_start_control_count=PREDICTION_HORIZON;
    std::copy_n(context.r1_nominal,PREDICTION_HORIZON,
                memory->warm_start_controls);
    const auto begin=std::chrono::steady_clock::now();
    const auto status=mpc_rti_solve_cycle(
        &context.initial_state,context.progress_m,context.trajectory.data(),
        context.trajectory.size(),context.lap_length_m,context.prediction_dt,
        PREDICTION_HORIZON,&config,memory.get(),result.get());
    const auto us=std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-begin).count();
    if(const char *folder=std::getenv("MPC_FROZEN_ELASTIC_CAPTURE_DIR");
       folder && repair>=11) {
      std::filesystem::create_directories(folder);
      const auto name=std::string("epoch_")+
          std::to_string(context.command_epoch_ns)+"_mode_"+
          std::to_string(repair);
      if(exported_first->valid)
        sdu_apex::qp_fixture::write_capture(
            (std::filesystem::path(folder)/(name+"_r1.fqp")).string(),
            *exported_first);
      if(exported_second->valid)
        sdu_apex::qp_fixture::write_capture(
            (std::filesystem::path(folder)/(name+"_r2.fqp")).string(),
            *exported_second);
    }
    std::cout << "{\"offline_normal_sqp_comparison\":\""
              << (repair==0?"repair_off":repair==1?"repair_on":
                  repair==2?"repair_feasibility_first":
                  repair==4?"r2_existing":
                  repair==6?"r2_no_progress":
                  repair==8?"rack_reachable":
                  repair==11?"r2_source_violation_filter":
                  repair==13?"elastic_penalty_100":
                  repair==14?"elastic_penalty_500":
                  repair==15?"elastic_penalty_1000":
                  repair==16?"elastic_penalty_4000":
                  repair==17?"elastic_rho_32_penalty_8000":
                  repair==18?"elastic_300_iter_penalty_8000":
                  repair==19?"elastic_adaptive_penalty_8000":
                  repair==20?"elastic_rho_32_300_iter_penalty_1000":
                  repair==21?"elastic_weight1000_rho127_tol002":
                  repair==22?"elastic_weight500_rho127_tol002":
                  repair==23?"elastic_weight8000_rho127_tol002":
                  repair==24?"elastic_weight1000_rho127_tol0005":
                  repair==25?"elastic_weight500_rho127_tol0005":
                  repair==27?"native_60_command_source_sqp_controller":
                  "elastic_default_precise_warm")
              << "\",\"status\":"
              << static_cast<int>(status)
              << ",\"source_nominal_repair_attempted\":"
              << result->source_nominal_repair_attempted
              << ",\"native_source_sqp_active\":"
              << result->native_source_sqp_active
              << ",\"native_source_sqp_qp_solves\":"
              << result->native_source_sqp_qp_solves
              << ",\"native_source_sqp_final_wall_violation_m2\":";
    json_number(result->native_source_sqp_final_wall_violation_m2);
    std::cout << ",\"native_source_sqp_first_feasible_iteration\":"
              << result->native_source_sqp_first_feasible_iteration
              << ",\"native_source_sqp_candidate_count\":"
              << result->candidate_control_count
              << ",\"source_nominal_repair_accepted\":"
              << result->source_nominal_repair_accepted
              << ",\"source_nominal_repair_iterations\":"
              << result->source_nominal_repair_phase_one_iterations;
    std::cout << ",\"sqp_completed\":"<<result->sqp_iterations_completed
              << ",\"refinement_nominal_us\":";
    json_number(result->refinement_nominal_build_us);
    std::cout << ",\"first_failure_reason\":"
              << result->line_search_full_step_failure_reason
              << ",\"first_failure_stage\":"
              << result->line_search_full_step_failure_stage;
    std::cout << ",\"selected_alpha\":";
    json_number(result->line_search_selected_alpha);
    std::cout << ",\"source_violation_filter_attempted\":"
              << result->source_violation_filter_attempted
              << ",\"source_violation_filter_selected\":"
              << result->source_violation_filter_selected
              << ",\"source_violation_filter_alpha\":";
    json_number(result->source_violation_filter_alpha);
    std::cout << ",\"source_violation_initial_m2\":";
    json_number(result->source_violation_initial_m2);
    std::cout << ",\"source_violation_selected_m2\":";
    json_number(result->source_violation_selected_m2);
    std::cout << ",\"source_nominal_repair_us\":";
    json_number(result->source_nominal_repair_us);
    std::cout << ",\"qp_admm_status\":"<<result->admm_status
              <<",\"qp_iterations\":"<<result->solver_iterations
              <<",\"qp_dual_residual\":";
    json_number(result->dual_residual);
    std::cout << ",\"r1_primal_residual\":";
    json_number(result->primal_residual);
    std::cout << ",\"nonlinear_failure_stage\":"
              << result->nonlinear_failure_stage
              << ",\"nonlinear_objective\":";
    json_number(result->nonlinear_objective);
    std::cout << ",\"full_cycle_us\":" << us << "}\n";
  }
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3 ||
      (argc == 3 && std::string(argv[2]) != "--source-phase1" &&
       std::string(argv[2]) != "--compare-normal-sqp" &&
       std::string(argv[2]) != "--native-source-sqp" &&
       std::string(argv[2]) != "--source-gradient" &&
       std::string(argv[2]) != "--qp-gradient" &&
       std::string(argv[2]) != "--clearance-gradient" &&
       std::string(argv[2]) != "--successor-probe" &&
       std::string(argv[2]) != "--sqp-directional" &&
       std::string(argv[2]) != "--sqp-directional-second-order" &&
       std::string(argv[2]) != "--sqp-cost-attribution" &&
       std::string(argv[2]) != "--export-progress-geometry" &&
       std::string(argv[2]) != "--progress-secant-audit" &&
       std::string(argv[2]).rfind("--external-controls=",0) != 0 &&
       std::string(argv[2]).rfind("--external-qp-decisions=",0) != 0 &&
       std::string(argv[2]).rfind("--external-qp-decisions-r2=",0) != 0)) {
    std::cerr << "usage: mpc_live_failure_replay CASE_DIRECTORY [--source-phase1|--compare-normal-sqp|--source-gradient|--qp-gradient|--clearance-gradient|--successor-probe|--sqp-directional]\n";
    return 2;
  }
  try {
    const std::filesystem::path directory(argv[1]);
    sdu_apex::qp_fixture::FailureReplayContext context{};
    if (!sdu_apex::qp_fixture::read_failure_context(
            (directory / "source_context.mctx").string(), &context))
      throw std::runtime_error("source context failed ABI, schema, or bounds checks");
    MpcRtiQpCapture_t r1{}, r2{};
    const bool has_r1 = sdu_apex::qp_fixture::read_capture(
        (directory / "r1.fqp").string(), &r1);
    const bool has_r2 = sdu_apex::qp_fixture::read_capture(
        (directory / "r2.fqp").string(), &r2);
    if (has_r1 != static_cast<bool>(context.r1_captured) ||
        has_r2 != static_cast<bool>(context.r2_captured))
      throw std::runtime_error("QP fixture and source-context passes differ");
    std::cout << std::setprecision(12);
    std::cout << "{\"context_schema\":" << context.format_version
              << ",\"command_epoch_ns\":"
              << context.command_epoch_ns << ",\"progress_m\":"
              << context.progress_m << ",\"trajectory_samples\":"
              << context.trajectory.size()
              << ",\"corridor_margin_m\":"<<context.model.corridor_margin_m
              << ",\"first_prediction_corridor_margin_m\":"<<context.model.first_prediction_corridor_margin_m
              << ",\"nonlinear_corridor_tolerance_m\":"<<context.model.nonlinear_corridor_tolerance_m
              << ",\"hidden_source_state_values\":"
              << MPC_PLANT_INTERNAL_STATE_COUNT
              << ",\"complete_actuator_queue\":["
              << context.initial_state.plant.delayed_steering_command_1 << ','
              << context.initial_state.plant.delayed_steering_command_2 << ','
              << context.initial_state.plant.delayed_throttle_command_1 << ','
              << context.initial_state.plant.delayed_throttle_command_2
              << "]}\n";
    if(argc==3 && std::string(argv[2])=="--native-source-sqp") {
      // The new backend must be replayable EVEN WHEN the old recorded
      // R1 raw QP cannot be exactly reproduced. Its independent full
      // source-origin context is valid and does not depend on the old QP.
      if(!has_r1)throw std::runtime_error(
          "native source SQP replay requires saved first command seed");
      emit_normal_sqp_comparison(context,r1,27);
      return 0;
    }
    if (has_r1)
      emit_pass("R1", r1, context.r1_nominal,
          context.r1_unscaled_proposal, context.r1_proposal,
          context.r1_controls_valid, context.r1_metrics, context);
    if (has_r1 && context.r1_controls_valid)
      emit_first_stage_recoverability(context, context.r1_nominal);
    if (has_r2)
      emit_pass("R2", r2, context.r2_nominal,
          context.r2_unscaled_proposal, context.r2_proposal,
          context.r2_controls_valid, context.r2_metrics, context);
    if (argc == 3 &&
        (std::string(argv[2]).rfind("--external-controls=",0)==0 ||
         std::string(argv[2]).rfind("--external-qp-decisions=",0)==0 ||
         std::string(argv[2]).rfind("--external-qp-decisions-r2=",0)==0)) {
      const bool second_qp=std::string(argv[2]).rfind("--external-qp-decisions-r2=",0)==0;
      const bool qp_decisions=second_qp ||
        std::string(argv[2]).rfind("--external-qp-decisions=",0)==0;
      if(second_qp && !has_r2)
        throw std::runtime_error("R2 QP not present in this capture");
      const auto *selected_capture=second_qp?&r2:&r1;
      auto external_capture=std::make_unique<MpcRtiQpCapture_t>();
      MpcRtiConfiguration_t qp_model=context.model;
      if(const char *file=std::getenv("MPC_EXTERNAL_REPLAY_QP_FIXTURE_PATH")) {
        if(!sdu_apex::qp_fixture::read_capture(file,external_capture.get()))
          throw std::runtime_error("independent full QP fixture invalid");
        selected_capture=external_capture.get();
        /* This opt-in experimental mode solves rack-angle-relative steering
         * increments, but retains the same physical source model. */
        qp_model.command_increment_qp_enabled=1;
        qp_model.source_reachable_steering_qp=1;
      }
      const auto &selected=*selected_capture;
      const MpcModelControl_t *baseline=
        std::getenv("MPC_EXTERNAL_REPLAY_QP_FIXTURE_PATH")
          ?selected.linearization_controls:
           second_qp?context.r2_nominal:context.r1_nominal;
      const auto equals=std::string(argv[2]).find('=');
      std::ifstream input(std::string(argv[2]).substr(equals+1));
      if(!input)throw std::runtime_error("external exact-source controls file missing");
      MpcModelControl_t candidates[PREDICTION_HORIZON]{};
      for(int k=0;k<PREDICTION_HORIZON;++k)
        if(!(input>>candidates[k].steering_target_rad>>
                     candidates[k].throttle_command))
          throw std::runtime_error("external exact-source control schema mismatch");
      if(qp_decisions) {
        float steering=selected.problem.x0[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND];
        float throttle=selected.problem.x0[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND];
        for(int k=0;k<PREDICTION_HORIZON;++k) {
          const auto &bounds=selected.problem.steps[k];
          const float dsteer=std::clamp(candidates[k].steering_target_rad,
                                     bounds.u_lb[0],bounds.u_ub[0]);
          const float dthrottle=std::clamp(candidates[k].throttle_command,
                                     bounds.u_lb[1],bounds.u_ub[1]);
          steering=context.model.command_increment_qp_enabled?
                steering+dsteer:dsteer;
          throttle=context.model.command_increment_qp_enabled?
                throttle+dthrottle:dthrottle;
          steering=std::clamp(steering,-context.model.max_steering_rad,
                                     context.model.max_steering_rad);
          throttle=std::clamp(throttle,MPC_THROTTLE_MIN_DRIVE,MPC_THROTTLE_MAX);
          candidates[k]={steering,throttle};
        }
      }
      const auto score=score_source_controls(context,candidates);
      const auto initial=score_source_controls(context,baseline);
      // Track individual native-unit errors; a single cross-unit maximum
      // conflates metres, radians, velocities, and wheel angular speeds.
      double coordinate_errors[MPC_RTI_NX]{};
      int source_state_comparison_valid=0;
      {
        auto wide=context.trajectory;
        for(auto &point:wide)point.left_bound=point.right_bound=100.f;
        MpcRtiCycleConfiguration_t validator{};
        validator.model=context.model;
        validator.model.world_sweep_check=nullptr;
        validator.model.world_sweep_context=nullptr;
        MpcRtiNominal_t source{},seed{};
        int failure=-1;
        if(mpc_rti_validate_candidate_plan(&context.initial_state,
              context.progress_m,candidates,wide.data(),wide.size(),
              context.lap_length_m,PREDICTION_HORIZON,context.prediction_dt,
              &validator,&source,&failure)==MPC_RTI_ROLLOUT_OK &&
           mpc_rti_validate_candidate_plan(&context.initial_state,
              context.progress_m,baseline,wide.data(),wide.size(),
              context.lap_length_m,PREDICTION_HORIZON,context.prediction_dt,
              &validator,&seed,&failure)==MPC_RTI_ROLLOUT_OK) {
          double predicted[MPC_RTI_NX]{};
          std::copy_n(selected.problem.x0,MPC_RTI_NX,predicted);
          for(int k=0;k<=PREDICTION_HORIZON;++k) {
            float exact[MPC_RTI_NX]{};
            serialize_state(&source.states[k],exact);
            exact[MPC_OPT_IDX_PROGRESS_DEVIATION]=float(
                source.progress[k]-seed.progress[k]);
            exact[MPC_OPT_IDX_PREVIOUS_STEERING_COMMAND]=
                source.states[k].plant.delayed_steering_command_2;
            exact[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]=
                source.states[k].plant.delayed_throttle_command_2;
            for(int j=0;j<MPC_RTI_NX;++j)
              coordinate_errors[j]=std::max(coordinate_errors[j],
                          std::abs(predicted[j]-double(exact[j])));
            if(k==PREDICTION_HORIZON)break;
            const auto &step=selected.problem.steps[k];
            const double command[2]={
                double(candidates[k].steering_target_rad)-
                  (qp_model.command_increment_qp_enabled?
                     predicted[mpc_rti_steering_qp_state_index(&qp_model)]:0.),
                double(candidates[k].throttle_command)-
                  (qp_model.command_increment_qp_enabled?
                     predicted[MPC_OPT_IDX_PREVIOUS_THROTTLE_COMMAND]:0.)};
            double next[MPC_RTI_NX]{};
            for(int j=0;j<MPC_RTI_NX;++j) {
              next[j]=step.d[j];
              for(int h=0;h<MPC_RTI_NX;++h)
                next[j]+=step.A[j][h]*predicted[h];
              for(int a=0;a<2;++a)next[j]+=step.B[j][a]*command[a];
            }
            std::copy_n(next,MPC_RTI_NX,predicted);
          }
          source_state_comparison_valid=1;
        }
      }

      MpcRtiSqpStepMetrics_t fidelity{};
      const bool fidelity_valid=mpc_rti_measure_qp_source_agreement(
        &context.initial_state,context.progress_m,baseline,candidates,
        &selected.problem,context.trajectory.data(),context.trajectory.size(),
        context.lap_length_m,context.prediction_dt,&qp_model,&fidelity);
      std::cout<<"{\"external_source_controls\":true,\"qp_pass\":\""
               <<(second_qp?"R2":"R1")
               <<"\",\"qp_source_fidelity_computed\":"
               <<(fidelity_valid?"true":"false")
               <<",\"qp_source_max_state_error\":";
      json_number(fidelity_valid?fidelity.maximum_state_error:NAN);
      std::cout<<",\"per_coordinate_prediction_valid\":"
               <<(source_state_comparison_valid?"true":"false")
               <<",\"per_coordinate_max_absolute_error\":[";
      for(int j=0;j<MPC_RTI_NX;++j) {
        if(j)std::cout<<',';
        json_number(source_state_comparison_valid?coordinate_errors[j]:NAN);
      }
      std::cout<<"]";
      std::cout<<",\"qp_source_max_state_error_stage\":"
               <<(fidelity_valid?fidelity.maximum_state_error_stage:-1)
               <<",\"qp_source_max_footprint_error_m\":";
      json_number(fidelity_valid?fidelity.maximum_footprint_constraint_error_m:NAN);
      std::cout<<",\"initial_valid\":"
               <<(initial.original_status==MPC_RTI_ROLLOUT_OK?"true":"false")
               <<",\"initial_failure_stage\":"<<initial.original_failure_stage
               <<",\"complete\":"
               <<(score.complete?"true":"false")
               <<",\"physical_valid\":"
               <<(score.original_status==MPC_RTI_ROLLOUT_OK?"true":"false")
               <<",\"failure_stage\":"<<score.original_failure_stage
               <<",\"source_objective\":";
      json_number(score.objective);
      std::cout<<",\"initial_objective\":";
      json_number(initial.objective);
      std::cout<<",\"source_improvement\":";
      json_number(score.complete&&initial.complete?
          initial.objective-score.objective:NAN);
      std::cout<<"}\n";
      for(double a:{.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625,.000244140625}) {
        MpcModelControl_t trial[PREDICTION_HORIZON]{};
        for(int k=0;k<PREDICTION_HORIZON;++k) {
          trial[k].steering_target_rad=float((1.-a)*baseline[k].steering_target_rad+
                                              a*candidates[k].steering_target_rad);
          trial[k].throttle_command=float((1.-a)*baseline[k].throttle_command+
                                         a*candidates[k].throttle_command);
        }
        const auto r=score_source_controls(context,trial);
        MpcRtiSqpStepMetrics_t scaled_metrics{};
        const bool scaled_fidelity=mpc_rti_measure_qp_source_agreement(
            &context.initial_state,context.progress_m,baseline,trial,
            &selected.problem,context.trajectory.data(),
            context.trajectory.size(),context.lap_length_m,
            context.prediction_dt,&context.model,&scaled_metrics);
        std::cout<<"{\"oracle_scaled_alpha\":"<<a
          <<",\"qp_pass\":\""<<(second_qp?"R2":"R1")<<"\""
          <<",\"valid\":"<<(r.original_status==MPC_RTI_ROLLOUT_OK?"true":"false")
          <<",\"failure_stage\":"<<r.original_failure_stage
          <<",\"max_source_state_error\":";
        json_number(scaled_fidelity?scaled_metrics.maximum_state_error:NAN);
        std::cout<<",\"max_source_footprint_error_m\":";
        json_number(scaled_fidelity?
            scaled_metrics.maximum_footprint_constraint_error_m:NAN);
        std::cout<<",\"source_improvement\":";
        json_number(r.complete&&initial.complete?initial.objective-r.objective:NAN);
        std::cout<<"}\n";
      }
    }
    if (argc == 3 && std::string(argv[2]) == "--progress-secant-audit")
      emit_progress_secant_audit(context);
    if (argc == 3 && std::string(argv[2]) == "--export-progress-geometry") {
      if(has_r1)emit_frozen_progress_geometry("R1",context,context.r1_nominal);
      if(has_r2)emit_frozen_progress_geometry("R2",context,context.r2_nominal);
    }
    if (argc == 3 && std::string(argv[2]) == "--sqp-cost-attribution") {
      if(context.r1_captured && context.r1_controls_valid)
        emit_sqp_cost_attribution("R1",context,
          context.r1_nominal,context.r1_proposal);
      if(context.r2_captured && context.r2_controls_valid)
        emit_sqp_cost_attribution("R2",context,
          context.r2_nominal,context.r2_proposal);
    }
    if (argc == 3 && std::string(argv[2]) ==
            "--sqp-directional-second-order") {
      if(context.r1_captured && context.r1_controls_valid)
        emit_sqp_directional_second_order("R1",context,r1,
          context.r1_nominal,context.r1_proposal);
      if(context.r2_captured && context.r2_controls_valid)
        emit_sqp_directional_second_order("R2",context,r2,
          context.r2_nominal,context.r2_proposal);
    }
    if (argc == 3 && std::string(argv[2]) == "--sqp-directional") {
      if(context.r1_captured && context.r1_controls_valid)
        emit_sqp_directional_consistency("R1",context,r1,
          context.r1_nominal,context.r1_proposal);
      if(context.r2_captured && context.r2_controls_valid)
        emit_sqp_directional_consistency("R2",context,r2,
          context.r2_nominal,context.r2_proposal);
    }
    if (argc == 3 && std::string(argv[2]) == "--successor-probe") {
      if (context.r1_captured)
        emit_source_successor_probe("R1",context,context.r1_nominal);
      if (context.r2_captured)
        emit_source_successor_probe("R2",context,context.r2_nominal);
    }
    if (argc == 3 && std::string(argv[2]) == "--clearance-gradient") {
      if (context.r1_captured)
        emit_clearance_gradient_probe("R1",context,r1,context.r1_nominal);
      if (context.r2_captured)
        emit_clearance_gradient_probe("R2",context,r2,context.r2_nominal);
    }
    if (argc == 3 && std::string(argv[2]) == "--qp-gradient") {
      if (context.r1_captured)
        emit_qp_gradient_probe("R1",context,r1,context.r1_nominal);
      if (context.r2_captured)
        emit_qp_gradient_probe("R2",context,r2,context.r2_nominal);
    }
    if (argc == 3 && std::string(argv[2]) == "--source-gradient") {
      if (context.r1_captured)
        emit_source_gradient_probe("R1_nominal",context,context.r1_nominal);
      if (context.r2_captured)
        emit_source_gradient_probe("R2_nominal",context,context.r2_nominal);
    }
    if (argc == 3 && std::string(argv[2]) == "--compare-normal-sqp" && has_r1)
      emit_normal_sqp_comparison(context,r1);
    if (argc == 3 && std::string(argv[2]) == "--source-phase1") {
      if (has_r1) emit_source_phase_one("R1_nominal", context, context.r1_nominal);
      if (has_r2) emit_source_phase_one("R2_nominal", context, context.r2_nominal);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "mpc_live_failure_replay: " << error.what() << '\n';
    return 2;
  }
}

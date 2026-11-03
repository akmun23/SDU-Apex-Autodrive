#pragma once

/* Same-build replay context for a rejected live SQP pass. The QP itself uses
 * the portable .fqp format; this companion file stores the source plant and
 * route PODs using an ABI/size guard so the exact nonlinear rollout can be
 * repeated by the matching build. */
#include "mpc_rti.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace sdu_apex::qp_fixture {

constexpr char kFailureContextMagic[8] = {'S','D','U','M','F','C','1','\n'};
constexpr std::uint32_t kFailureContextVersion = 6;

struct FailureReplayContext {
  std::uint32_t format_version{};
  std::int64_t command_epoch_ns{};
  double progress_m{};
  double lap_length_m{};
  float prediction_dt{};
  int terminal_viability{};
  int world_sweep_was_configured{};
  MpcRtiState_t initial_state{};
  MpcRtiConfiguration_t model{};
  MpcRtiObjectiveSchedule_t objective{};
  MpcRtiSqpStepMetrics_t r1_metrics{};
  MpcRtiSqpStepMetrics_t r2_metrics{};
  MpcModelControl_t r1_nominal[PREDICTION_HORIZON]{};
  MpcModelControl_t r1_unscaled_proposal[PREDICTION_HORIZON]{};
  MpcModelControl_t r1_proposal[PREDICTION_HORIZON]{};
  MpcModelControl_t r1_source_rollout[PREDICTION_HORIZON]{};
  MpcModelControl_t r2_nominal[PREDICTION_HORIZON]{};
  MpcModelControl_t r2_unscaled_proposal[PREDICTION_HORIZON]{};
  MpcModelControl_t r2_proposal[PREDICTION_HORIZON]{};
  MpcModelControl_t r2_source_rollout[PREDICTION_HORIZON]{};
  int r1_captured{};
  int r2_captured{};
  int r1_controls_valid{};
  int r2_controls_valid{};
  int r1_source_rollout_controls_valid{};
  int r2_source_rollout_controls_valid{};
  std::vector<MpcTrajectorySample_t> trajectory;
};

static_assert(std::is_trivially_copyable<MpcRtiState_t>::value,
              "failure replay requires POD plant state");
static_assert(std::is_trivially_copyable<MpcRtiConfiguration_t>::value,
              "failure replay requires POD model configuration");
static_assert(std::is_trivially_copyable<MpcRtiObjectiveSchedule_t>::value,
              "failure replay requires POD objective schedule");
static_assert(std::is_trivially_copyable<MpcRtiSqpStepMetrics_t>::value,
              "failure replay requires POD step metrics");
static_assert(std::is_trivially_copyable<MpcTrajectorySample_t>::value,
              "failure replay requires POD trajectory samples");

inline bool write_failure_context(const std::string &path,
                                  const FailureReplayContext &context) {
  if (context.trajectory.size() < 3 || context.trajectory.size() > 1000000 ||
      !context.objective.valid || context.objective.horizon != PREDICTION_HORIZON)
    return false;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.good()) return false;
  const std::uint32_t version = kFailureContextVersion;
  const std::uint32_t trajectory_count =
      static_cast<std::uint32_t>(context.trajectory.size());
  const std::uint32_t state_size = sizeof(MpcRtiState_t);
  const std::uint32_t model_size = sizeof(MpcRtiConfiguration_t);
  const std::uint32_t objective_size = sizeof(MpcRtiObjectiveSchedule_t);
  const std::uint32_t metric_size = sizeof(MpcRtiSqpStepMetrics_t);
  const std::uint32_t sample_size = sizeof(MpcTrajectorySample_t);
  const auto bytes = [&out](const auto &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
  };
  out.write(kFailureContextMagic, sizeof(kFailureContextMagic));
  bytes(version); bytes(state_size); bytes(model_size); bytes(objective_size);
  bytes(metric_size); bytes(sample_size); bytes(trajectory_count);
  bytes(context.command_epoch_ns); bytes(context.progress_m);
  bytes(context.lap_length_m); bytes(context.prediction_dt);
  bytes(context.terminal_viability); bytes(context.world_sweep_was_configured);
  bytes(context.initial_state); bytes(context.model); bytes(context.objective);
  bytes(context.r1_metrics); bytes(context.r2_metrics);
  bytes(context.r1_nominal); bytes(context.r1_unscaled_proposal);
  bytes(context.r1_proposal); bytes(context.r1_source_rollout);
  bytes(context.r2_nominal); bytes(context.r2_unscaled_proposal);
  bytes(context.r2_proposal); bytes(context.r2_source_rollout);
  bytes(context.r1_captured); bytes(context.r2_captured);
  bytes(context.r1_controls_valid); bytes(context.r2_controls_valid);
  bytes(context.r1_source_rollout_controls_valid);
  bytes(context.r2_source_rollout_controls_valid);
  out.write(reinterpret_cast<const char *>(context.trajectory.data()),
            static_cast<std::streamsize>(context.trajectory.size() *
                                         sizeof(MpcTrajectorySample_t)));
  return out.good();
}

inline bool read_failure_context(const std::string &path,
                                 FailureReplayContext *context) {
  if (!context) return false;
  *context = FailureReplayContext{};
  std::ifstream in(path, std::ios::binary);
  char magic[sizeof(kFailureContextMagic)]{};
  std::uint32_t version=0, state_size=0, model_size=0, objective_size=0;
  std::uint32_t metric_size=0, sample_size=0, trajectory_count=0;
  if (!in.good()) return false;
  in.read(magic, sizeof(magic));
  const auto read = [&in](auto *value) {
    in.read(reinterpret_cast<char *>(value), sizeof(*value));
  };
  read(&version); read(&state_size); read(&model_size); read(&objective_size);
  read(&metric_size); read(&sample_size); read(&trajectory_count);
  if (!in.good() || std::memcmp(magic, kFailureContextMagic, sizeof(magic)) ||
      (version != 1 && version != 2 && version != 3 && version != 4 &&
       version != 5 && version != kFailureContextVersion) ||
      state_size != sizeof(MpcRtiState_t) ||
      (version<5
        ? model_size != offsetof(MpcRtiConfiguration_t,weight_source_lateral_slip) &&
          model_size != offsetof(MpcRtiConfiguration_t,source_reachable_steering_qp) &&
          model_size != sizeof(MpcRtiConfiguration_t)
        : version==5
          ? model_size != offsetof(MpcRtiConfiguration_t,source_reachable_steering_qp) &&
            model_size != sizeof(MpcRtiConfiguration_t)
          : model_size != sizeof(MpcRtiConfiguration_t)) ||
      (version<4
        ? objective_size != offsetof(MpcRtiObjectiveSchedule_t, successor_recourse_valid)
        : objective_size != sizeof(MpcRtiObjectiveSchedule_t)) ||
      metric_size != sizeof(MpcRtiSqpStepMetrics_t) ||
      sample_size != sizeof(MpcTrajectorySample_t) ||
      trajectory_count < 3 || trajectory_count > 1000000) return false;
  context->format_version = version;
  read(&context->command_epoch_ns); read(&context->progress_m);
  read(&context->lap_length_m); read(&context->prediction_dt);
  read(&context->terminal_viability); read(&context->world_sweep_was_configured);
  read(&context->initial_state);
  in.read(reinterpret_cast<char *>(&context->model),model_size);
  if(version<5)context->model.weight_source_lateral_slip=0.f;
  if(version<6)context->model.source_reachable_steering_qp=0;
  in.read(reinterpret_cast<char *>(&context->objective),objective_size);
  if(version<4) {
    context->objective.successor_recourse_valid=0;
    context->objective.successor_recourse_weight=0.f;
  }
  /* v1/v2 model ABI ended at world_map_rejection_only. The newly appended
   * reserve cost occupies the former alignment padding, whose serialized
   * bytes were NOT meaningful physical configuration. Never interpret those
   * bytes as an opt-in cost during historic source-exact replay. */
  if(version < 3) context->model.weight_source_corridor_reserve = 0.f;
  read(&context->r1_metrics); read(&context->r2_metrics);
  read(&context->r1_nominal); read(&context->r1_unscaled_proposal);
  read(&context->r1_proposal); read(&context->r1_source_rollout);
  read(&context->r2_nominal); read(&context->r2_unscaled_proposal);
  read(&context->r2_proposal); read(&context->r2_source_rollout);
  if (version == 1) {
    read(&context->r1_controls_valid); read(&context->r2_controls_valid);
    context->r1_captured = context->r1_controls_valid;
    context->r2_captured = context->r2_controls_valid;
    context->r1_source_rollout_controls_valid = context->r1_controls_valid;
    context->r2_source_rollout_controls_valid = context->r2_controls_valid;
  } else {
    read(&context->r1_captured); read(&context->r2_captured);
    read(&context->r1_controls_valid); read(&context->r2_controls_valid);
    read(&context->r1_source_rollout_controls_valid);
    read(&context->r2_source_rollout_controls_valid);
  }
  if (!in.good()) return false;
  context->trajectory.resize(trajectory_count);
  in.read(reinterpret_cast<char *>(context->trajectory.data()),
          static_cast<std::streamsize>(trajectory_count *
                                       sizeof(MpcTrajectorySample_t)));
  if (!in.good() || in.peek() != std::char_traits<char>::eof() ||
      !context->objective.valid ||
      context->objective.horizon != PREDICTION_HORIZON ||
      !std::isfinite(context->progress_m) ||
      !std::isfinite(context->lap_length_m) || context->lap_length_m <= 0.0 ||
      !std::isfinite(context->prediction_dt) || context->prediction_dt <= 0.0f ||
      context->terminal_viability < 0 || context->terminal_viability > 1 ||
      context->r1_captured < 0 || context->r1_captured > 1 ||
      context->r2_captured < 0 || context->r2_captured > 1 ||
      context->r1_controls_valid < 0 || context->r1_controls_valid > 1 ||
      context->r2_controls_valid < 0 || context->r2_controls_valid > 1 ||
      context->r1_source_rollout_controls_valid < 0 ||
      context->r1_source_rollout_controls_valid > 1 ||
      context->r2_source_rollout_controls_valid < 0 ||
      context->r2_source_rollout_controls_valid > 1)
    return false;
  return true;
}

}  // namespace sdu_apex::qp_fixture

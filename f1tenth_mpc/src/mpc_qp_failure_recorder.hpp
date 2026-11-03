#pragma once
// Opt-in forensic recorder: never performs filesystem I/O on the 40 Hz
// controller callback. It is not enabled in production profiles.
#include "mpc_qp_fixture_format.hpp"
#include "mpc_qp_failure_context.hpp"
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace f1tenth_mpc {
class MpcQpFailureRecorder {
 public:
  explicit MpcQpFailureRecorder(std::filesystem::path directory)
      : directory_(std::move(directory)) {
    if (directory_.empty())
      throw std::invalid_argument("failure capture directory is empty");
    std::filesystem::create_directories(directory_);
    thread_ = std::thread([this] { run(); });
  }
  MpcQpFailureRecorder(const MpcQpFailureRecorder &) = delete;
  MpcQpFailureRecorder &operator=(const MpcQpFailureRecorder &) = delete;

  ~MpcQpFailureRecorder() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

  void submit(std::int64_t command_epoch_ns, double progress_m,
              float speed_ceiling_mps, const MpcRtiState_t &initial,
              MpcRtiCycleStatus_t status, const MpcRtiCycleResult_t &result,
              const MpcRtiCycleConfiguration_t &configuration,
              const MpcTrajectorySample_t *trajectory,
              std::size_t trajectory_count, double lap_length_m,
              float prediction_dt,
              const MpcRtiQpCapture_t *r1,
              const MpcRtiQpCapture_t *r2) {
    if ((!r1 || !r1->valid) && (!r2 || !r2->valid)) return;
    if (!trajectory || trajectory_count < 3 ||
        trajectory_count > 1000000) return;
    // Allocate and copy only on a failed/abnormal controller cycle, never
    // during the normal accepted 40 Hz hot path.
    auto snapshot = std::make_unique<Snapshot>();
    snapshot->epoch = command_epoch_ns;
    snapshot->progress = progress_m;
    snapshot->ceiling = speed_ceiling_mps;
    snapshot->initial = initial;
    snapshot->status = status;
    snapshot->admm_status = result.admm_status;
    snapshot->primal = result.primal_residual;
    snapshot->dual = result.dual_residual;
    snapshot->failure_stage = result.nonlinear_failure_stage;
    snapshot->failure_reason = result.nonlinear_failure_reason;
    snapshot->nominal_count = std::clamp(result.nominal_control_count,
                                         0, PREDICTION_HORIZON);
    for (int i = 0; i < PREDICTION_HORIZON; ++i)
      snapshot->nominal[i] = result.nominal_controls[i];
    auto &source = snapshot->source;
    source.command_epoch_ns = command_epoch_ns;
    source.progress_m = progress_m;
    source.lap_length_m = lap_length_m;
    source.prediction_dt = prediction_dt;
    source.terminal_viability = configuration.terminal_viability;
    source.world_sweep_was_configured =
        configuration.model.world_sweep_check != nullptr;
    source.initial_state = initial;
    source.model = configuration.model;
    source.model.world_sweep_check = nullptr;
    source.model.world_sweep_context = nullptr;
    source.objective = result.comparison_objective;
    source.r1_metrics = result.r1_step_metrics;
    source.r2_metrics = result.r2_step_metrics;
    source.trajectory.assign(trajectory, trajectory + trajectory_count);
    if (r1 && r1->valid) {
      source.r1_captured = 1;
      std::copy_n(r1->linearization_controls, PREDICTION_HORIZON,
                  source.r1_nominal);
      std::copy_n(r1->unscaled_proposal_controls, PREDICTION_HORIZON,
                  source.r1_unscaled_proposal);
      std::copy_n(r1->proposal_controls, PREDICTION_HORIZON,
                  source.r1_proposal);
      std::copy_n(r1->source_rollout_controls, PREDICTION_HORIZON,
                  source.r1_source_rollout);
      source.r1_controls_valid = r1->proposal_controls_valid;
      source.r1_source_rollout_controls_valid =
          r1->source_rollout_controls_valid;
    }
    if (r2 && r2->valid) {
      source.r2_captured = 1;
      std::copy_n(r2->linearization_controls, PREDICTION_HORIZON,
                  source.r2_nominal);
      std::copy_n(r2->unscaled_proposal_controls, PREDICTION_HORIZON,
                  source.r2_unscaled_proposal);
      std::copy_n(r2->proposal_controls, PREDICTION_HORIZON,
                  source.r2_proposal);
      std::copy_n(r2->source_rollout_controls, PREDICTION_HORIZON,
                  source.r2_source_rollout);
      source.r2_controls_valid = r2->proposal_controls_valid;
      source.r2_source_rollout_controls_valid =
          r2->source_rollout_controls_valid;
    }
    if (r1 && r1->valid)
      snapshot->first = std::make_unique<MpcRtiQpCapture_t>(*r1);
    if (r2 && r2->valid)
      snapshot->second = std::make_unique<MpcRtiQpCapture_t>(*r2);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_ || queued_.size() >= 4) return;
      snapshot->serial = next_serial_++;
      queued_.push_back(std::move(snapshot));
    }
    ready_.notify_one();
  }

 private:
  struct Snapshot {
    std::uint64_t serial{};
    std::int64_t epoch{};
    double progress{};
    float ceiling{};
    MpcRtiState_t initial{};
    MpcRtiCycleStatus_t status{};
    int admm_status{};
    float primal{}, dual{};
    int failure_stage{}, failure_reason{};
    int nominal_count{};
    MpcModelControl_t nominal[PREDICTION_HORIZON]{};
    sdu_apex::qp_fixture::FailureReplayContext source{};
    std::unique_ptr<MpcRtiQpCapture_t> first, second;
  };

  static void write_csv(const std::filesystem::path &base,
                        const Snapshot &snapshot) {
    const auto &p = snapshot.initial.plant;
    std::ofstream state(base / "physical_state.csv");
    state << "command_epoch_ns,progress_m,speed_ceiling_mps,e_y,e_psi,u,v,r,"
             "actual_steering_rad,delayed_steering_1,delayed_steering_2,"
             "delayed_throttle_1,delayed_throttle_2,plant_initialized,"
             "suspension_history_initialized,imu_state_initialized,"
             "incoming_branch_flags,incoming_transition_valid";
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 6; ++j)
        state << ",incoming_tracking_jacobian_" << i << '_' << j;
    for (int i = 0; i < 4; ++i)
      state << ",incoming_wheel_slip_excess_" << i;
    for (int i = 0; i < MPC_PLANT_INTERNAL_STATE_COUNT; ++i)
      state << ",source_internal_" << i;
    state << '\n' << std::setprecision(17)
          << snapshot.epoch << ',' << snapshot.progress << ','
          << snapshot.ceiling << ',' << p.e_y << ',' << p.e_psi << ','
          << p.u << ',' << p.v << ',' << p.r << ','
          << p.actual_steering_angle << ','
          << p.delayed_steering_command_1 << ','
          << p.delayed_steering_command_2 << ','
          << p.delayed_throttle_command_1 << ','
          << p.delayed_throttle_command_2 << ','
          << p.source_plant_initialized << ','
          << p.source_suspension_history_initialized << ','
          << p.source_imu_state_initialized << ','
          << snapshot.source.initial_state.incoming_branch_flags << ','
          << snapshot.source.initial_state.incoming_transition_valid;
    for (const auto &row : snapshot.source.initial_state.incoming_tracking_jacobian)
      for (float value : row) state << ',' << value;
    for (double value : snapshot.source.initial_state.incoming_wheel_slip_excess_mps)
      state << ',' << value;
    for (double x : p.source_plant_internal_state) state << ',' << x;
    state << '\n';
    std::ofstream nominal(base / "nominal_source_seed.csv");
    nominal << std::setprecision(17) << 0;
    for (int k = 0; k < snapshot.nominal_count; ++k)
      nominal << ',' << snapshot.nominal[k].steering_target_rad
              << ',' << snapshot.nominal[k].throttle_command;
    nominal << '\n';
  }

  void write(const Snapshot &snapshot) const {
    const auto base = directory_ /
        ("epoch_" + std::to_string(snapshot.epoch) +
         "_case_" + std::to_string(snapshot.serial));
    std::filesystem::create_directories(base);
    if (snapshot.first &&
        !sdu_apex::qp_fixture::write_capture(
            (base / "r1.fqp").string(), *snapshot.first))
      throw std::runtime_error("R1 QP fixture write failed");
    if (snapshot.second &&
        !sdu_apex::qp_fixture::write_capture(
            (base / "r2.fqp").string(), *snapshot.second))
      throw std::runtime_error("R2 QP fixture write failed");
    if (!sdu_apex::qp_fixture::write_failure_context(
            (base / "source_context.mctx").string(), snapshot.source))
      throw std::runtime_error("source replay context write failed");
    write_csv(base, snapshot);
    std::ofstream metadata(base / "context.json");
    metadata << std::setprecision(17)
             << "{\"command_epoch_ns\":" << snapshot.epoch
             << ",\"progress_m\":" << snapshot.progress
             << ",\"active_speed_ceiling_mps\":" << snapshot.ceiling
             << ",\"cycle_status\":" << snapshot.status
             << ",\"admm_status\":" << snapshot.admm_status
             << ",\"primal_residual\":" << snapshot.primal
             << ",\"dual_residual\":" << snapshot.dual
             << ",\"nonlinear_failure_stage\":" << snapshot.failure_stage
             << ",\"nonlinear_failure_reason\":" << snapshot.failure_reason
             << ",\"r1_captured\":" << (snapshot.first ? "true" : "false")
             << ",\"r2_captured\":" << (snapshot.second ? "true" : "false")
             << "}\n";
    if (!metadata.good())
      throw std::runtime_error("QP failure metadata write failed");
  }

  void run() {
    while (true) {
      std::unique_ptr<Snapshot> current;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stopping_ || !queued_.empty(); });
        if (queued_.empty()) return;
        current = std::move(queued_.front());
        queued_.pop_front();
      }
      try {
        write(*current);
      } catch (const std::exception &e) {
        std::cerr << "MPC forensic QP capture failed: " << e.what() << '\n';
      }
    }
  }

  std::filesystem::path directory_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::unique_ptr<Snapshot>> queued_;
  std::thread thread_;
  std::uint64_t next_serial_{};
  bool stopping_{};
};
}  // namespace f1tenth_mpc

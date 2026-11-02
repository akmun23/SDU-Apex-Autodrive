#pragma once
// Opt-in forensic recorder: never performs filesystem I/O on the 40 Hz
// controller callback. It is not enabled in production profiles.
#include "mpc_qp_fixture_format.hpp"
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
              const MpcRtiQpCapture_t *r1,
              const MpcRtiQpCapture_t *r2) {
    if ((!r1 || !r1->valid) && (!r2 || !r2->valid)) return;
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
    std::unique_ptr<MpcRtiQpCapture_t> first, second;
  };

  static void write_csv(const std::filesystem::path &base,
                        const Snapshot &snapshot) {
    const auto &p = snapshot.initial.plant;
    std::ofstream state(base / "physical_state.csv");
    state << std::setprecision(17)
          << snapshot.epoch << ',' << snapshot.progress << ','
          << snapshot.ceiling << ','
          << p.e_y << ',' << p.e_psi << ',' << p.u << ','
          << p.v << ',' << p.r << ',' << p.actual_steering_angle << ','
          << p.delayed_steering_command_1 << ','
          << p.delayed_steering_command_2 << ','
          << p.delayed_throttle_command_1 << ','
          << p.delayed_throttle_command_2 << ','
          << p.source_suspension_history_initialized << ",0,0";
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

#ifndef MPC_CONTROL_TIME_PREDICTOR_HPP
#define MPC_CONTROL_TIME_PREDICTOR_HPP

#include "mpc_reference.h"
#include "mpc_state_synchronizer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace f1tenth_mpc {

constexpr std::size_t kMpcCommandHistoryCapacity = 16;

struct MpcCommandHistoryEntry {
    int64_t stamp_ns{};
    double steering_command_rad{};
    double target_speed_mps{};
};

/* Fixed-capacity causal command history. Entries must be strictly ordered by
 * their ROS header timestamp; overflow replaces the oldest entry. */
class MpcCommandHistory {
public:
    bool push(const MpcCommandHistoryEntry &entry);
    void clear();
    std::size_t size() const;
    bool at(std::size_t chronological_index,
            MpcCommandHistoryEntry *entry) const;
    bool latest_at_or_before(int64_t stamp_ns,
                             MpcCommandHistoryEntry *entry) const;
    bool next_after(int64_t stamp_ns, int64_t target_stamp_ns,
                    MpcCommandHistoryEntry *entry) const;

private:
    std::array<MpcCommandHistoryEntry, kMpcCommandHistoryCapacity> entries_{};
    std::size_t oldest_index_{};
    std::size_t size_{};
};

enum class MpcControlTimeMode {
    kNoExtrapolation,
    kConstantBodyTwist,
    kAcceptedModelCommandHistory,
};

enum class MpcControlTimeStatus {
    kOk,
    kInvalidInput,
    kTargetBeforeSource,
    kStateTooOld,
    kProjectionFailed,
    kInvalidModelStep,
};

struct MpcControlTimePredictorConfig {
    double maximum_state_age_s{0.120};
    double model_integration_step_s{0.025};
    double maximum_command_speed_mps{16.0};
};

struct MpcControlTimePrediction {
    MpcSynchronizedState state{};
    MpcPathProjection_t projection{};
    double progress_m{};
    double age_s{};
    double target_speed_mps{};
    double steering_command_rad{};
    double previous_steering_rate_radps{};
    double previous_target_speed_rate_mps2{};
    std::size_t command_changes_used{};
    bool used_command_fallback{};
    bool used_time_fallback{};
};

MpcControlTimeStatus predict_to_control_time(
    MpcControlTimeMode mode,
    const MpcSynchronizedState &source_state,
    int64_t target_stamp_ns,
    const MpcCommandHistory &command_history,
    const MpcTrajectorySample_t *trajectory,
    std::size_t trajectory_count,
    double lap_length_m,
    std::size_t previous_segment,
    const MpcPathProjection_t *known_source_projection,
    const MpcControlTimePredictorConfig &config,
    MpcControlTimePrediction *prediction);

const char *control_time_status_name(MpcControlTimeStatus status);

}  // namespace f1tenth_mpc

#endif

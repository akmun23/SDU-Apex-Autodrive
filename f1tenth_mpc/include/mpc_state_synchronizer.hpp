#ifndef MPC_STATE_SYNCHRONIZER_HPP
#define MPC_STATE_SYNCHRONIZER_HPP

#include <cstddef>
#include <cstdint>
#include <deque>

namespace f1tenth_mpc {

struct MpcOdomSample {
    int64_t stamp_ns{};
    double x{};
    double y{};
    double yaw{};
    double u{};
    double v{};
    double yaw_rate{};
};

struct MpcMapPoseAnchor {
    int64_t stamp_ns{};
    double x{};
    double y{};
    double yaw{};
};

struct MpcSynchronizedState {
    int64_t source_stamp_ns{};
    int64_t odom_source_stamp_ns{};
    int64_t map_pose_source_stamp_ns{};
    double map_x{};
    double map_y{};
    double map_yaw{};
    double u{};
    double v{};
    double yaw_rate{};
    double source_age_s{};
    double pose_odom_skew_s{};
};

enum class MpcSyncStatus {
    kOk,
    kMissingOdom,
    kMissingMapPose,
    kInvalidInput,
    kTimestampOrderFault,
    kSourceGapFault,
    kNoOdomBracket,
    kMapPoseFutureOfOdom,
    kPoseOdomSkewExceeded,
    kStateTooOld,
    kCommandTimeBeforeState,
};

struct MpcSyncConfig {
    double source_dt_min_s{0.001};
    double source_dt_max_s{0.250};
    double max_pose_odom_skew_s{0.120};
    double max_state_age_s{0.120};
    std::size_t odom_buffer_capacity{64};
};

class MpcStateSynchronizer {
public:
    explicit MpcStateSynchronizer(const MpcSyncConfig &config = {});

    MpcSyncStatus push_odometry(const MpcOdomSample &sample);
    MpcSyncStatus set_map_pose(const MpcMapPoseAnchor &anchor);
    MpcSyncStatus synchronize(int64_t command_time_ns,
                              MpcSynchronizedState *state) const;
    void reset();
    static const char *status_name(MpcSyncStatus status);

private:
    MpcSyncConfig config_;
    std::deque<MpcOdomSample> odometry_;
    MpcMapPoseAnchor map_pose_{};
    bool map_pose_valid_{false};
    MpcSyncStatus latched_fault_{MpcSyncStatus::kOk};
};

}  // namespace f1tenth_mpc

#endif

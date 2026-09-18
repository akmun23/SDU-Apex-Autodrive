#include "mpc_state_synchronizer.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace f1tenth_mpc {
namespace {

constexpr double kNsToSeconds = 1.0e-9;

double wrap_angle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

bool finite_odom(const MpcOdomSample &sample)
{
    return sample.stamp_ns > 0 && std::isfinite(sample.x) &&
        std::isfinite(sample.y) && std::isfinite(sample.yaw) &&
        std::isfinite(sample.u) && std::isfinite(sample.v) &&
        std::isfinite(sample.yaw_rate);
}

bool finite_pose(const MpcMapPoseAnchor &pose)
{
    return pose.stamp_ns > 0 && std::isfinite(pose.x) &&
        std::isfinite(pose.y) && std::isfinite(pose.yaw);
}

bool interpolate_odom(const std::deque<MpcOdomSample> &samples,
                      int64_t stamp_ns, MpcOdomSample *output)
{
    if (!output || samples.empty() || stamp_ns < samples.front().stamp_ns ||
        stamp_ns > samples.back().stamp_ns) return false;

    const auto upper = std::lower_bound(samples.begin(), samples.end(), stamp_ns,
        [](const MpcOdomSample &sample, int64_t stamp) {
            return sample.stamp_ns < stamp;
        });
    if (upper == samples.end()) return false;
    if (upper->stamp_ns == stamp_ns || upper == samples.begin()) {
        *output = *upper;
        return true;
    }

    const auto lower = std::prev(upper);
    const double alpha = static_cast<double>(stamp_ns - lower->stamp_ns) /
        static_cast<double>(upper->stamp_ns - lower->stamp_ns);
    *output = *lower;
    output->stamp_ns = stamp_ns;
    output->x += alpha * (upper->x - lower->x);
    output->y += alpha * (upper->y - lower->y);
    output->yaw = wrap_angle(lower->yaw + alpha *
        wrap_angle(upper->yaw - lower->yaw));
    output->u += alpha * (upper->u - lower->u);
    output->v += alpha * (upper->v - lower->v);
    output->yaw_rate += alpha * (upper->yaw_rate - lower->yaw_rate);
    return true;
}

}  // namespace

MpcStateSynchronizer::MpcStateSynchronizer(MpcSyncConfig config)
    : config_(config)
{
    if (config_.odom_buffer_capacity < 2) {
        config_ = MpcSyncConfig{};
    }
}

MpcSyncStatus MpcStateSynchronizer::push_odometry(const MpcOdomSample &sample)
{
    if (!finite_odom(sample)) return MpcSyncStatus::kInvalidInput;
    if (!odometry_.empty() && sample.stamp_ns <= odometry_.back().stamp_ns) {
        // Keep the newly received sample and rebase the interpolation window.
        // Arrival jitter, duplicate stamps, and source-clock reversals are
        // diagnostics, not reasons to latch the controller in neutral.
        odometry_.clear();
    }
    odometry_.push_back(sample);
    while (odometry_.size() > config_.odom_buffer_capacity)
        odometry_.pop_front();
    return MpcSyncStatus::kOk;
}

MpcSyncStatus MpcStateSynchronizer::set_map_pose(
    const MpcMapPoseAnchor &anchor)
{
    if (!finite_pose(anchor)) return MpcSyncStatus::kInvalidInput;
    // The newest callback is the best available localization result even if
    // its source stamp is older than a pose already delivered to this node.
    map_pose_ = anchor;
    map_pose_valid_ = true;
    return MpcSyncStatus::kOk;
}

MpcSyncStatus MpcStateSynchronizer::synchronize(
    int64_t command_time_ns, MpcSynchronizedState *state) const
{
    if (!state) return MpcSyncStatus::kInvalidInput;
    if (odometry_.empty()) return MpcSyncStatus::kMissingOdom;
    if (!map_pose_valid_) return MpcSyncStatus::kMissingMapPose;

    const MpcOdomSample &latest = odometry_.back();
    MpcOdomSample odom_at_anchor;
    if (!interpolate_odom(odometry_, map_pose_.stamp_ns, &odom_at_anchor)) {
        // A pose can arrive outside the retained odometry window under host
        // jitter. Use the nearest available odometry endpoint instead of
        // rejecting the state; this preserves a finite best-effort estimate.
        const MpcOdomSample &front = odometry_.front();
        odom_at_anchor =
            std::llabs(map_pose_.stamp_ns - front.stamp_ns) <
                    std::llabs(map_pose_.stamp_ns - latest.stamp_ns) ?
            front : latest;
    }

    const double dx_odom = latest.x - odom_at_anchor.x;
    const double dy_odom = latest.y - odom_at_anchor.y;
    const double c_odom = std::cos(odom_at_anchor.yaw);
    const double s_odom = std::sin(odom_at_anchor.yaw);
    const double relative_x = c_odom * dx_odom + s_odom * dy_odom;
    const double relative_y = -s_odom * dx_odom + c_odom * dy_odom;
    const double c_map = std::cos(map_pose_.yaw);
    const double s_map = std::sin(map_pose_.yaw);

    const int64_t fused_stamp_ns = std::max(latest.stamp_ns, map_pose_.stamp_ns);
    const double age_s = static_cast<double>(command_time_ns - fused_stamp_ns) *
        kNsToSeconds;

    state->source_stamp_ns = fused_stamp_ns;
    state->map_x = map_pose_.x + c_map * relative_x - s_map * relative_y;
    state->map_y = map_pose_.y + s_map * relative_x + c_map * relative_y;
    state->map_yaw = wrap_angle(map_pose_.yaw +
                                wrap_angle(latest.yaw - odom_at_anchor.yaw));
    state->u = latest.u;
    state->v = latest.v;
    state->yaw_rate = latest.yaw_rate;
    state->source_age_s = age_s;
    state->pose_odom_skew_s = static_cast<double>(
        latest.stamp_ns - map_pose_.stamp_ns) * kNsToSeconds;
    return MpcSyncStatus::kOk;
}

void MpcStateSynchronizer::reset()
{
    odometry_.clear();
    map_pose_ = MpcMapPoseAnchor{};
    map_pose_valid_ = false;
    latched_fault_ = MpcSyncStatus::kOk;
}

const char *MpcStateSynchronizer::status_name(MpcSyncStatus status)
{
    switch (status) {
    case MpcSyncStatus::kOk: return "ok";
    case MpcSyncStatus::kMissingOdom: return "missing_odometry";
    case MpcSyncStatus::kMissingMapPose: return "missing_map_pose";
    case MpcSyncStatus::kInvalidInput: return "invalid_input";
    case MpcSyncStatus::kTimestampOrderFault: return "timestamp_order_fault";
    case MpcSyncStatus::kSourceGapFault: return "source_gap_fault";
    case MpcSyncStatus::kNoOdomBracket: return "no_odometry_bracket";
    case MpcSyncStatus::kMapPoseFutureOfOdom: return "map_pose_future_of_odometry";
    case MpcSyncStatus::kPoseOdomSkewExceeded: return "pose_odom_skew_exceeded";
    case MpcSyncStatus::kStateTooOld: return "state_too_old";
    case MpcSyncStatus::kCommandTimeBeforeState: return "command_time_before_state";
    }
    return "unknown";
}

}  // namespace f1tenth_mpc

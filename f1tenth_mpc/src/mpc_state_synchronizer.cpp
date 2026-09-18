#include "mpc_state_synchronizer.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace f1tenth_mpc {
namespace {

constexpr double kNsToSeconds = 1.0e-9;
constexpr double kTimestampQuantizationToleranceS = 1.0e-6;

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
    if (!(config_.source_dt_min_s > 0.0) ||
        !(config_.source_dt_max_s >= config_.source_dt_min_s) ||
        !(config_.max_pose_odom_skew_s > 0.0) ||
        !(config_.max_state_age_s > 0.0) ||
        config_.odom_buffer_capacity < 2) {
        config_ = MpcSyncConfig{};
    }
}

MpcSyncStatus MpcStateSynchronizer::push_odometry(const MpcOdomSample &sample)
{
    if (latched_fault_ != MpcSyncStatus::kOk) return latched_fault_;
    if (!finite_odom(sample)) return MpcSyncStatus::kInvalidInput;
    if (!odometry_.empty()) {
        const int64_t delta_ns = sample.stamp_ns - odometry_.back().stamp_ns;
        if (delta_ns <= 0) {
            latched_fault_ = MpcSyncStatus::kTimestampOrderFault;
            return latched_fault_;
        }
        const double dt = static_cast<double>(delta_ns) * kNsToSeconds;
        /* Simulator source time is exported through a floating-point clock;
         * tolerate one microsecond of endpoint quantization without changing
         * or resampling the measured interval. */
        if (dt + kTimestampQuantizationToleranceS < config_.source_dt_min_s ||
            dt - kTimestampQuantizationToleranceS > config_.source_dt_max_s) {
            latched_fault_ = MpcSyncStatus::kSourceGapFault;
            return latched_fault_;
        }
    }
    odometry_.push_back(sample);
    while (odometry_.size() > config_.odom_buffer_capacity)
        odometry_.pop_front();
    return MpcSyncStatus::kOk;
}

MpcSyncStatus MpcStateSynchronizer::set_map_pose(
    const MpcMapPoseAnchor &anchor)
{
    if (latched_fault_ != MpcSyncStatus::kOk) return latched_fault_;
    if (!finite_pose(anchor)) return MpcSyncStatus::kInvalidInput;
    if (map_pose_valid_ && anchor.stamp_ns < map_pose_.stamp_ns) {
        latched_fault_ = MpcSyncStatus::kTimestampOrderFault;
        return latched_fault_;
    }
    map_pose_ = anchor;
    map_pose_valid_ = true;
    return MpcSyncStatus::kOk;
}

MpcSyncStatus MpcStateSynchronizer::synchronize(
    int64_t command_time_ns, MpcSynchronizedState *state) const
{
    if (!state) return MpcSyncStatus::kInvalidInput;
    if (latched_fault_ != MpcSyncStatus::kOk) return latched_fault_;
    if (odometry_.empty()) return MpcSyncStatus::kMissingOdom;
    if (!map_pose_valid_) return MpcSyncStatus::kMissingMapPose;

    const MpcOdomSample &latest = odometry_.back();
    if (command_time_ns < latest.stamp_ns)
        return MpcSyncStatus::kCommandTimeBeforeState;
    if (map_pose_.stamp_ns > latest.stamp_ns)
        return MpcSyncStatus::kMapPoseFutureOfOdom;

    const double skew_s = static_cast<double>(latest.stamp_ns - map_pose_.stamp_ns) *
        kNsToSeconds;
    if (skew_s > config_.max_pose_odom_skew_s)
        return MpcSyncStatus::kPoseOdomSkewExceeded;

    MpcOdomSample odom_at_anchor;
    if (!interpolate_odom(odometry_, map_pose_.stamp_ns, &odom_at_anchor))
        return MpcSyncStatus::kNoOdomBracket;

    const double dx_odom = latest.x - odom_at_anchor.x;
    const double dy_odom = latest.y - odom_at_anchor.y;
    const double c_odom = std::cos(odom_at_anchor.yaw);
    const double s_odom = std::sin(odom_at_anchor.yaw);
    const double relative_x = c_odom * dx_odom + s_odom * dy_odom;
    const double relative_y = -s_odom * dx_odom + c_odom * dy_odom;
    const double c_map = std::cos(map_pose_.yaw);
    const double s_map = std::sin(map_pose_.yaw);

    const double age_s = static_cast<double>(command_time_ns - latest.stamp_ns) *
        kNsToSeconds;
    if (age_s > config_.max_state_age_s) return MpcSyncStatus::kStateTooOld;

    state->source_stamp_ns = latest.stamp_ns;
    state->map_x = map_pose_.x + c_map * relative_x - s_map * relative_y;
    state->map_y = map_pose_.y + s_map * relative_x + c_map * relative_y;
    state->map_yaw = wrap_angle(map_pose_.yaw +
                                wrap_angle(latest.yaw - odom_at_anchor.yaw));
    state->u = latest.u;
    state->v = latest.v;
    state->yaw_rate = latest.yaw_rate;
    state->source_age_s = age_s;
    state->pose_odom_skew_s = skew_s;
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

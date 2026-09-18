#include "mpc_control_time_predictor.hpp"

#include "mpc_types.h"
#include "vehicle_model.h"

#include <algorithm>
#include <cmath>

namespace f1tenth_mpc {
namespace {

constexpr double kNsToSeconds = 1.0e-9;
constexpr double kAgeToleranceSeconds = 1.0e-9;

double wrap_angle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

bool finite_source(const MpcSynchronizedState &state)
{
    return state.source_stamp_ns > 0 && std::isfinite(state.map_x) &&
        std::isfinite(state.map_y) && std::isfinite(state.map_yaw) &&
        std::isfinite(state.u) && std::isfinite(state.v) &&
        std::isfinite(state.yaw_rate);
}

bool project_source(const MpcSynchronizedState &source,
                    const MpcTrajectorySample_t *trajectory,
                    std::size_t trajectory_count, double lap_length_m,
                    std::size_t previous_segment,
                    MpcPathProjection_t *projection)
{
    return mpc_trajectory_project(trajectory, trajectory_count, lap_length_m,
        source.map_x, source.map_y, source.map_yaw, previous_segment, 160,
        projection) != 0;
}

bool project_predicted_map_pose(MpcControlTimePrediction *prediction,
                                const MpcTrajectorySample_t *trajectory,
                                std::size_t trajectory_count,
                                double lap_length_m,
                                std::size_t previous_segment)
{
    if (!mpc_trajectory_project(trajectory, trajectory_count, lap_length_m,
            prediction->state.map_x, prediction->state.map_y,
            prediction->state.map_yaw, previous_segment, 160,
            &prediction->projection)) return false;
    prediction->progress_m = prediction->projection.s;
    return true;
}

bool propagate_model_segment(MpcModelState_t *state, double *progress_m,
                             int64_t start_ns, int64_t end_ns,
                             double integration_step_s,
                             const MpcTrajectorySample_t *trajectory,
                             std::size_t trajectory_count,
                             double lap_length_m)
{
    if (end_ns < start_ns) return false;
    int64_t cursor_ns = start_ns;
    while (cursor_ns < end_ns) {
        const double remaining_s = static_cast<double>(end_ns - cursor_ns) *
            kNsToSeconds;
        const float dt = static_cast<float>(
            std::min(integration_step_s, remaining_s));
        MpcTrajectorySample_t sample{};
        if (!(dt > 0.0f) || !mpc_trajectory_sample(trajectory,
                trajectory_count, lap_length_m, *progress_m, &sample))
            return false;
        const MpcModelControl_t held_command_rate{0.0f, 0.0f};
        const MpcStageResult_t result = mpc_vehicle_model_step(
            state, &held_command_rate, dt, static_cast<float>(sample.curvature));
        if (!result.valid) return false;
        *state = result.next;
        *progress_m += result.delta_s_m;
        const int64_t step_ns = std::min<int64_t>(end_ns - cursor_ns,
            std::max<int64_t>(1, static_cast<int64_t>(std::llround(
                static_cast<double>(dt) * 1.0e9))));
        cursor_ns += step_ns;
    }
    return true;
}

}  // namespace

bool MpcCommandHistory::push(const MpcCommandHistoryEntry &entry)
{
    if (entry.stamp_ns <= 0 || !std::isfinite(entry.steering_command_rad) ||
        !std::isfinite(entry.target_speed_mps) ||
        entry.target_speed_mps < 0.0 ||
        entry.target_speed_mps > MPC_MAX_COMMAND_SPEED_MPS ||
        std::abs(entry.steering_command_rad) > SOURCE_MAX_STEERING_RAD)
        return false;
    if (size_ > 0) {
        MpcCommandHistoryEntry newest{};
        if (!at(size_ - 1, &newest) || entry.stamp_ns <= newest.stamp_ns)
            return false;
    }

    if (size_ < entries_.size()) {
        entries_[(oldest_index_ + size_) % entries_.size()] = entry;
        ++size_;
    } else {
        entries_[oldest_index_] = entry;
        oldest_index_ = (oldest_index_ + 1) % entries_.size();
    }
    return true;
}

void MpcCommandHistory::clear()
{
    oldest_index_ = 0;
    size_ = 0;
}

std::size_t MpcCommandHistory::size() const
{
    return size_;
}

bool MpcCommandHistory::at(std::size_t chronological_index,
                           MpcCommandHistoryEntry *entry) const
{
    if (!entry || chronological_index >= size_) return false;
    *entry = entries_[(oldest_index_ + chronological_index) % entries_.size()];
    return true;
}

bool MpcCommandHistory::latest_at_or_before(
    int64_t stamp_ns, MpcCommandHistoryEntry *entry) const
{
    if (!entry) return false;
    bool found = false;
    for (std::size_t i = 0; i < size_; ++i) {
        MpcCommandHistoryEntry candidate{};
        if (!at(i, &candidate)) return false;
        if (candidate.stamp_ns > stamp_ns) break;
        *entry = candidate;
        found = true;
    }
    return found;
}

bool MpcCommandHistory::next_after(int64_t stamp_ns, int64_t target_stamp_ns,
                                   MpcCommandHistoryEntry *entry) const
{
    if (!entry || target_stamp_ns < stamp_ns) return false;
    for (std::size_t i = 0; i < size_; ++i) {
        MpcCommandHistoryEntry candidate{};
        if (!at(i, &candidate)) return false;
        if (candidate.stamp_ns > stamp_ns) {
            if (candidate.stamp_ns > target_stamp_ns) return false;
            *entry = candidate;
            return true;
        }
    }
    return false;
}

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
    MpcControlTimePrediction *prediction)
{
    const bool valid_mode = mode == MpcControlTimeMode::kNoExtrapolation ||
        mode == MpcControlTimeMode::kConstantBodyTwist ||
        mode == MpcControlTimeMode::kAcceptedModelCommandHistory;
    if (!prediction || !finite_source(source_state) || target_stamp_ns <= 0 ||
        !valid_mode ||
        !trajectory || trajectory_count < 3 || !std::isfinite(lap_length_m) ||
        !(lap_length_m > 0.0) || !std::isfinite(config.maximum_state_age_s) ||
        config.maximum_state_age_s < 0.0 ||
        !std::isfinite(config.model_integration_step_s) ||
        !(config.model_integration_step_s > 0.0) ||
        !std::isfinite(config.maximum_command_speed_mps) ||
        config.maximum_command_speed_mps <= 0.0 ||
        config.maximum_command_speed_mps > MPC_MAX_COMMAND_SPEED_MPS)
        return MpcControlTimeStatus::kInvalidInput;
    if (target_stamp_ns < source_state.source_stamp_ns)
        return MpcControlTimeStatus::kTargetBeforeSource;

    const double age_s = static_cast<double>(target_stamp_ns -
        source_state.source_stamp_ns) * kNsToSeconds;
    if (age_s > config.maximum_state_age_s + kAgeToleranceSeconds)
        return MpcControlTimeStatus::kStateTooOld;

    MpcControlTimePrediction result{};
    result.state = source_state;
    result.age_s = age_s;
    result.state.source_age_s = age_s;
    MpcCommandHistoryEntry command_at_target{};
    if (command_history.latest_at_or_before(target_stamp_ns,
            &command_at_target)) {
        result.target_speed_mps = command_at_target.target_speed_mps;
        result.steering_command_rad = command_at_target.steering_command_rad;
        MpcCommandHistoryEntry previous_command{};
        for (std::size_t i = 1; i < command_history.size(); ++i) {
            MpcCommandHistoryEntry candidate{};
            MpcCommandHistoryEntry following{};
            if (!command_history.at(i - 1, &candidate) ||
                !command_history.at(i, &following)) break;
            if (following.stamp_ns > target_stamp_ns) break;
            previous_command = candidate;
        }
        if (previous_command.stamp_ns > 0 &&
            command_at_target.stamp_ns > previous_command.stamp_ns) {
            const double command_dt_s = static_cast<double>(
                command_at_target.stamp_ns - previous_command.stamp_ns) *
                kNsToSeconds;
            result.previous_steering_rate_radps =
                (command_at_target.steering_command_rad -
                 previous_command.steering_command_rad) / command_dt_s;
            result.previous_target_speed_rate_mps2 =
                (command_at_target.target_speed_mps -
                 previous_command.target_speed_mps) / command_dt_s;
        }
    } else {
        result.target_speed_mps = std::clamp(source_state.u, 0.0,
            config.maximum_command_speed_mps);
    }
    if (known_source_projection) {
        if (!std::isfinite(known_source_projection->s) ||
            !std::isfinite(known_source_projection->lateral_error) ||
            !std::isfinite(known_source_projection->heading_error) ||
            known_source_projection->segment >= trajectory_count)
            return MpcControlTimeStatus::kInvalidInput;
        result.projection = *known_source_projection;
    } else if (!project_source(source_state, trajectory, trajectory_count,
                   lap_length_m, previous_segment, &result.projection)) {
        return MpcControlTimeStatus::kProjectionFailed;
    }
    result.progress_m = result.projection.s;

    if (mode == MpcControlTimeMode::kConstantBodyTwist && age_s > 0.0) {
        const double midpoint_yaw = source_state.map_yaw +
            0.5 * source_state.yaw_rate * age_s;
        result.state.map_x += age_s * (source_state.u * std::cos(midpoint_yaw) -
            source_state.v * std::sin(midpoint_yaw));
        result.state.map_y += age_s * (source_state.u * std::sin(midpoint_yaw) +
            source_state.v * std::cos(midpoint_yaw));
        result.state.map_yaw = wrap_angle(source_state.map_yaw +
            source_state.yaw_rate * age_s);
        if (!project_predicted_map_pose(&result, trajectory, trajectory_count,
                lap_length_m, result.projection.segment))
            return MpcControlTimeStatus::kProjectionFailed;
    } else if (mode == MpcControlTimeMode::kAcceptedModelCommandHistory) {
        MpcModelState_t plant{};
        plant.e_y = static_cast<float>(result.projection.lateral_error);
        plant.e_psi = static_cast<float>(result.projection.heading_error);
        plant.u = static_cast<float>(std::max(0.0, source_state.u));
        plant.v = static_cast<float>(source_state.v);
        plant.r = static_cast<float>(source_state.yaw_rate);
        MpcCommandHistoryEntry active_command{};
        if (command_history.latest_at_or_before(source_state.source_stamp_ns,
                &active_command)) {
            plant.target_speed = static_cast<float>(active_command.target_speed_mps);
            plant.steering_command =
                static_cast<float>(active_command.steering_command_rad);
        } else {
            plant.target_speed = static_cast<float>(std::clamp(source_state.u,
                0.0, config.maximum_command_speed_mps));
            result.used_command_fallback = true;
        }
        result.target_speed_mps = plant.target_speed;
        result.steering_command_rad = plant.steering_command;

        int64_t cursor_ns = source_state.source_stamp_ns;
        std::size_t guard = 0;
        while (cursor_ns < target_stamp_ns) {
            MpcCommandHistoryEntry next_command{};
            const bool has_next = command_history.next_after(cursor_ns,
                target_stamp_ns, &next_command);
            const int64_t segment_end_ns = has_next ? next_command.stamp_ns :
                target_stamp_ns;
            if (!propagate_model_segment(&plant, &result.progress_m, cursor_ns,
                    segment_end_ns, config.model_integration_step_s, trajectory,
                    trajectory_count, lap_length_m))
                return MpcControlTimeStatus::kInvalidModelStep;
            cursor_ns = segment_end_ns;
            if (has_next) {
                plant.target_speed = static_cast<float>(std::clamp(
                    next_command.target_speed_mps, 0.0,
                    config.maximum_command_speed_mps));
                plant.steering_command = static_cast<float>(
                    next_command.steering_command_rad);
                result.target_speed_mps = plant.target_speed;
                result.steering_command_rad = plant.steering_command;
                ++result.command_changes_used;
            } else {
                break;
            }
            if (++guard > kMpcCommandHistoryCapacity)
                return MpcControlTimeStatus::kInvalidInput;
        }

        MpcTrajectorySample_t final_sample{};
        if (!mpc_trajectory_sample(trajectory, trajectory_count, lap_length_m,
                result.progress_m, &final_sample))
            return MpcControlTimeStatus::kInvalidModelStep;
        const double integrated_progress_m = result.progress_m;
        result.state.map_x = final_sample.x -
            std::sin(final_sample.heading) * plant.e_y;
        result.state.map_y = final_sample.y +
            std::cos(final_sample.heading) * plant.e_y;
        result.state.map_yaw = wrap_angle(final_sample.heading + plant.e_psi);
        result.state.u = plant.u;
        result.state.v = plant.v;
        result.state.yaw_rate = plant.r;
        if (!project_predicted_map_pose(&result, trajectory, trajectory_count,
                lap_length_m, result.projection.segment))
            return MpcControlTimeStatus::kProjectionFailed;
        /* Keep the accepted model's integrated along-track state; projection
         * is wrapped and used only for local segment/cross-track diagnostics. */
        result.progress_m = integrated_progress_m;
    }

    *prediction = result;
    return MpcControlTimeStatus::kOk;
}

const char *control_time_status_name(MpcControlTimeStatus status)
{
    switch (status) {
    case MpcControlTimeStatus::kOk: return "ok";
    case MpcControlTimeStatus::kInvalidInput: return "invalid_input";
    case MpcControlTimeStatus::kTargetBeforeSource: return "target_before_source";
    case MpcControlTimeStatus::kStateTooOld: return "state_too_old";
    case MpcControlTimeStatus::kProjectionFailed: return "projection_failed";
    case MpcControlTimeStatus::kInvalidModelStep: return "invalid_model_step";
    }
    return "unknown";
}

}  // namespace f1tenth_mpc

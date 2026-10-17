#include "mpc_control_time_predictor.hpp"

#include "mpc_types.h"
#include "vehicle_model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace f1tenth_mpc {
namespace {

constexpr double kNsToSeconds = 1.0e-9;
constexpr int64_t kUnityPhysicsTickNs = 1000000;

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
                             int64_t grid_origin_ns,
                             int64_t *integrated_elapsed_ns,
                             double integration_step_s,
                             const MpcTrajectorySample_t *trajectory,
                             std::size_t trajectory_count,
                             double lap_length_m,
                             double throttle_normalized)
{
    if (end_ns < start_ns || !integrated_elapsed_ns ||
        start_ns < grid_origin_ns || end_ns < grid_origin_ns) return false;

    /* The source vehicle advances on Unity's fixed 1 ms physics grid. Bridge
     * header stamps are receive-time stamps and commonly contain fractional
     * milliseconds, so passing the raw final remainder to the plant makes it
     * reject an otherwise valid rollout. Quantize event boundaries relative
     * to this prediction's source stamp and preserve that phase across command
     * segments; this discards at most one sub-millisecond tail for the whole
     * propagation, not one tail per command event. */
    const int64_t target_ticks = (end_ns - grid_origin_ns) /
        kUnityPhysicsTickNs;
    int64_t integrated_ticks = *integrated_elapsed_ns /
        kUnityPhysicsTickNs;
    if (target_ticks < integrated_ticks) return false;
    const int64_t configured_ticks = static_cast<int64_t>(std::llround(
        integration_step_s * 1000.0));
    if (configured_ticks <= 0) return false;

    while (integrated_ticks < target_ticks) {
        const int64_t step_ticks = std::min(
            configured_ticks, target_ticks - integrated_ticks);
        const float dt = static_cast<float>(step_ticks) * 0.001f;
        MpcTrajectorySample_t sample{};
        if (!(dt > 0.0f) || !mpc_trajectory_sample(trajectory,
                trajectory_count, lap_length_m, *progress_m, &sample))
            return false;
        const MpcModelControl_t held_command_rate{
            0.0f, static_cast<float>(std::clamp(
                throttle_normalized, 0.0, 1.0))};
        const MpcStageResult_t result = mpc_vehicle_model_step(
            state, &held_command_rate, dt, static_cast<float>(sample.curvature));
        if (!result.valid) return false;
        *state = result.next;
        *progress_m += result.delta_s_m;
        integrated_ticks += step_ticks;
        *integrated_elapsed_ns = integrated_ticks * kUnityPhysicsTickNs;
    }
    return true;
}

}  // namespace

bool MpcCommandHistory::push(const MpcCommandHistoryEntry &entry)
{
    if (entry.stamp_ns <= 0 || !std::isfinite(entry.steering_command_rad) ||
        std::abs(entry.steering_command_rad) > SOURCE_MAX_STEERING_RAD ||
        (entry.throttle_valid && (!std::isfinite(entry.throttle_normalized) ||
            entry.throttle_normalized < 0.0 || entry.throttle_normalized > 1.0)))
        return false;
    if (size_ > 0) {
        MpcCommandHistoryEntry newest{};
        if (!at(size_ - 1, &newest)) return false;
        if (entry.stamp_ns == newest.stamp_ns) {
            entries_[(oldest_index_ + size_ - 1) % entries_.size()] = entry;
            return true;
        }
        if (entry.stamp_ns < newest.stamp_ns) clear();
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
    const MpcModelState_t *source_plant_state,
    int64_t source_plant_stamp_ns,
    double source_plant_progress_m,
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
        !std::isfinite(config.command_actuation_delay_s) ||
        config.command_actuation_delay_s < 0.0 ||
        (source_plant_state && (source_plant_stamp_ns <= 0 ||
            source_plant_stamp_ns > target_stamp_ns ||
            !std::isfinite(source_plant_progress_m))))
        return MpcControlTimeStatus::kInvalidInput;
    const double age_s = static_cast<double>(target_stamp_ns -
        source_state.source_stamp_ns) * kNsToSeconds;

    MpcControlTimePrediction result{};
    result.state = source_state;
    result.age_s = age_s;
    result.state.source_age_s = age_s;
    /* The bridge can preserve the nominal 40 Hz sensor stream while queued
     * actuator requests reach Unity later. Shift command events to their
     * effective application time, but leave the measured sensor state and its
     * receive timestamp untouched. This keeps the runtime causal and uses no
     * simulator-only feedback. */
    MpcCommandHistory effective_command_history;
    const MpcCommandHistory *causal_command_history = &command_history;
    if (config.command_actuation_delay_s > 0.0) {
        const int64_t delay_ns = static_cast<int64_t>(std::llround(
            config.command_actuation_delay_s * 1.0e9));
        for (std::size_t i = 0; i < command_history.size(); ++i) {
            MpcCommandHistoryEntry entry{};
            if (!command_history.at(i, &entry) || delay_ns < 0 ||
                entry.stamp_ns > std::numeric_limits<int64_t>::max() - delay_ns ||
                !effective_command_history.push({
                    entry.stamp_ns + delay_ns,
                    entry.steering_command_rad,
                    entry.throttle_normalized,
                    entry.throttle_valid})) {
                return MpcControlTimeStatus::kInvalidInput;
            }
        }
        causal_command_history = &effective_command_history;
    }
    const auto record_command_event = [&result](
        const MpcCommandHistoryEntry &entry) {
        if (entry.stamp_ns <= 0 ||
            result.command_event_stamp_count >=
                result.command_event_stamps_ns.size()) return;
        if (result.command_event_stamp_count > 0 &&
            result.command_event_stamps_ns[
                result.command_event_stamp_count - 1] == entry.stamp_ns) return;
        result.command_event_stamps_ns[
            result.command_event_stamp_count++] = entry.stamp_ns;
    };
    MpcCommandHistoryEntry command_at_target{};
    if (causal_command_history->latest_at_or_before(target_stamp_ns,
            &command_at_target)) {
        result.steering_command_rad = command_at_target.steering_command_rad;
        if (command_at_target.throttle_valid)
            result.throttle_normalized = command_at_target.throttle_normalized;
        MpcCommandHistoryEntry previous_command{};
        for (std::size_t i = 1; i < causal_command_history->size(); ++i) {
            MpcCommandHistoryEntry candidate{};
            MpcCommandHistoryEntry following{};
            if (!causal_command_history->at(i - 1, &candidate) ||
                !causal_command_history->at(i, &following)) break;
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
        }
    }
    result.previous_throttle_command = source_plant_state ?
        std::clamp(static_cast<double>(
            source_plant_state->delayed_throttle_command_2), 0.0, 1.0) :
        result.throttle_normalized;
    if (command_at_target.throttle_valid)
        result.previous_throttle_command = command_at_target.throttle_normalized;
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

    if (age_s < 0.0 || age_s > config.maximum_state_age_s) {
        // Do not turn source/arrival jitter into a command inhibit. For a
        // state outside the predictor's validated support, retain the newest
        // legal state without extrapolating it; the measured age remains in
        // diagnostics and the MPC still receives a usable state.
        result.used_time_fallback = true;
        if (causal_command_history->latest_at_or_before(target_stamp_ns,
                &command_at_target)) {
            result.steering_command_rad =
                command_at_target.steering_command_rad;
            if (command_at_target.throttle_valid)
                result.throttle_normalized = command_at_target.throttle_normalized;
            if (command_at_target.throttle_valid)
                result.previous_throttle_command =
                    command_at_target.throttle_normalized;
            record_command_event(command_at_target);
        } else {
            result.used_command_fallback = true;
        }
        *prediction = result;
        return MpcControlTimeStatus::kOk;
    }

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
        MpcModelState_t plant = source_plant_state ? *source_plant_state :
            MpcModelState_t{};
        double throttle_normalized = source_plant_state ?
            std::clamp(static_cast<double>(plant.delayed_throttle_command_2), 0.0, 1.0) :
            result.throttle_normalized;
        const int64_t plant_stamp_ns = source_plant_state ?
            source_plant_stamp_ns : source_state.source_stamp_ns;
        const double initial_progress_m = source_plant_state ?
            source_plant_progress_m : result.projection.s;
        result.plant_source_stamp_ns = plant_stamp_ns;
        result.plant_to_sensor_s = static_cast<double>(
            source_state.source_stamp_ns - plant_stamp_ns) * kNsToSeconds;
        result.progress_m = initial_progress_m;

        /* Keep a newly received encoder sample aside until the model reaches
         * that sample's timestamp. Applying it to the previous nominal state
         * would move a future measurement backwards in time. */
        const bool wheel_feedback_valid =
            plant.source_rear_wheel_omega_feedback_valid != 0;
        const float wheel_feedback[2] = {
            plant.source_rear_wheel_omega_feedback_radps[0],
            plant.source_rear_wheel_omega_feedback_radps[1]};
        plant.source_rear_wheel_omega_feedback_valid = 0;

        if (!source_plant_state || (!plant.source_plant_initialized &&
                plant_stamp_ns == source_state.source_stamp_ns)) {
            plant.e_y = static_cast<float>(result.projection.lateral_error);
            plant.e_psi = static_cast<float>(result.projection.heading_error);
            plant.u = static_cast<float>(std::max(0.0, source_state.u));
            plant.v = static_cast<float>(source_state.v);
            plant.r = static_cast<float>(source_state.yaw_rate);
        }

        MpcCommandHistoryEntry active_command{};
        if (causal_command_history->latest_at_or_before(
                plant_stamp_ns, &active_command)) {
            record_command_event(active_command);
            plant.steering_command = static_cast<float>(
                active_command.steering_command_rad);
            if (active_command.throttle_valid)
                throttle_normalized = active_command.throttle_normalized;
        } else if (!source_plant_state) {
            result.used_command_fallback = true;
        }
        result.steering_command_rad = plant.steering_command;
        result.throttle_normalized = throttle_normalized;

        int64_t cursor_ns = plant_stamp_ns;
        int64_t integrated_elapsed_ns = 0;
        std::size_t guard = 0;
        auto propagate_until = [&](int64_t end_ns) {
            while (cursor_ns < end_ns) {
                MpcCommandHistoryEntry next_command{};
                const bool has_next = causal_command_history->next_after(
                    cursor_ns, end_ns, &next_command);
                const int64_t segment_end_ns = has_next ?
                    next_command.stamp_ns : end_ns;
                if (!propagate_model_segment(&plant, &result.progress_m,
                        cursor_ns, segment_end_ns, plant_stamp_ns,
                        &integrated_elapsed_ns,
                        config.model_integration_step_s, trajectory,
                        trajectory_count, lap_length_m,
                        throttle_normalized)) return false;
                cursor_ns = segment_end_ns;
                if (!has_next) break;
                record_command_event(next_command);
                plant.steering_command = static_cast<float>(
                    next_command.steering_command_rad);
                if (next_command.throttle_valid)
                    throttle_normalized = next_command.throttle_normalized;
                result.steering_command_rad = plant.steering_command;
                result.throttle_normalized = throttle_normalized;
                ++result.command_changes_used;
                if (++guard > kMpcCommandHistoryCapacity) return false;
            }
            return true;
        };

        if (source_state.source_stamp_ns >= plant_stamp_ns) {
            if (!propagate_until(source_state.source_stamp_ns))
                return MpcControlTimeStatus::kInvalidModelStep;

            /* Re-anchor observable body state at its own source timestamp,
             * while retaining the propagated hidden wheel/suspension and
             * actuator history. The wrapped track coordinate is unwrapped
             * nearest the propagated progress to preserve lap count. */
            const double unwrapped_observed_progress = result.projection.s +
                std::round((result.progress_m - result.projection.s) /
                    lap_length_m) * lap_length_m;
            result.progress_m = unwrapped_observed_progress;
            plant.e_y = static_cast<float>(result.projection.lateral_error);
            plant.e_psi = static_cast<float>(result.projection.heading_error);
            plant.u = static_cast<float>(std::max(0.0, source_state.u));
            plant.v = static_cast<float>(source_state.v);
            plant.r = static_cast<float>(source_state.yaw_rate);
            if (wheel_feedback_valid) {
                plant.source_rear_wheel_omega_feedback_radps[0] =
                    wheel_feedback[0];
                plant.source_rear_wheel_omega_feedback_radps[1] =
                    wheel_feedback[1];
                if (plant.source_plant_initialized) {
                    plant.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RL] =
                        wheel_feedback[0];
                    plant.source_plant_state[UNITY_SOURCE_WHEEL_OMEGA_RR] =
                        wheel_feedback[1];
                    plant.source_rear_wheel_omega_feedback_valid = 0;
                } else {
                    /* The first model step will seed Unity's hidden plant from
                     * the observed body state and consume these wheel speeds. */
                    plant.source_rear_wheel_omega_feedback_valid = 1;
                }
            }
            result.sensor_state_fused = true;
        }

        /* Persist the hidden plant at the newest measurement epoch, not at
         * the future control epoch. At 40 Hz, a legal observation commonly
         * arrives with a source stamp about 5 ms before the previous
         * control-time prediction. Carrying the latter makes every following
         * observation appear retroactive and silently disables state fusion.
         * The caller carries this snapshot to the next callback; plant_state
         * below remains the separate copy advanced to the control instant. */
        result.plant_handoff_state = plant;
        result.plant_handoff_state_valid = true;
        result.plant_handoff_stamp_ns = cursor_ns;
        result.plant_handoff_progress_m = result.progress_m;

        if (!propagate_until(target_stamp_ns))
            return MpcControlTimeStatus::kInvalidModelStep;

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
        result.plant_state = plant;
        result.plant_state_valid = true;
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

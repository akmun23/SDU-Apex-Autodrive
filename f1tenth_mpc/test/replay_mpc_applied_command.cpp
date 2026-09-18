#define _POSIX_C_SOURCE 200809L

#include "mpc_control_time_predictor.hpp"
#include "vehicle_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#ifndef MPC_APPLIED_REPLAY_TRAJECTORY_PATH
#error "MPC_APPLIED_REPLAY_TRAJECTORY_PATH must name the accepted raceline"
#endif

using namespace f1tenth_mpc;

namespace {

constexpr std::size_t kMaximumTrajectoryPoints = 4000;
constexpr int64_t kControlStepNs = 25'000'000LL;
constexpr std::array<int, 5> kHorizonSteps{{1, 5, 10, 20, 30}};
constexpr std::array<const char *, 6> kSpeedBinNames{{"0-0.5", "0.5-2", "2-4",
                                                       "4-6", "6-8", ">8"}};
constexpr std::array<const char *, 7> kDimensionNames{{"map_position_m",
    "along_track_m", "cross_track_m", "yaw_rad", "u_mps", "v_mps",
    "yaw_rate_radps"}};

struct Event {
    int64_t arrival_ns{};
    int64_t header_stamp_ns{};
    std::string topic;
    std::string payload;
    std::size_t input_order{};
};

struct Bucket {
    std::vector<double> errors;
};

void fail(const std::string &message)
{
    std::cerr << "applied-command replay error: " << message << '\n';
    std::exit(2);
}

std::vector<std::string> split_csv(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(field);
    return fields;
}

bool parse_number(const std::string &text, double *value)
{
    if (!value) return false;
    char *end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || !std::isfinite(parsed))
        return false;
    *value = parsed;
    return true;
}

bool json_number(const std::string &json, const char *key, double *value)
{
    const std::string token = std::string("\"") + key + "\"";
    const std::size_t position = json.find(token);
    if (position == std::string::npos) return false;
    const std::size_t colon = json.find(':', position + token.size());
    if (colon == std::string::npos) return false;
    char *end = nullptr;
    const double parsed = std::strtod(json.c_str() + colon + 1, &end);
    if (end == json.c_str() + colon + 1 || !std::isfinite(parsed)) return false;
    *value = parsed;
    return true;
}

std::vector<Event> load_events(const std::string &path)
{
    std::ifstream input(path);
    if (!input.good()) fail("cannot open event stream " + path);
    std::vector<Event> events;
    std::string line;
    std::size_t order = 0;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        if (fields.size() < 8 || fields[0] == "event_index") continue;
        char *arrival_end = nullptr;
        char *stamp_end = nullptr;
        const int64_t arrival = std::strtoll(fields[2].c_str(), &arrival_end, 10);
        const int64_t stamp = std::strtoll(fields[5].c_str(), &stamp_end, 10);
        if (arrival_end == fields[2].c_str() || *arrival_end != '\0') continue;
        int64_t header_stamp = 0;
        if (stamp_end != fields[5].c_str() && *stamp_end == '\0')
            header_stamp = stamp;
        events.push_back(Event{arrival, header_stamp, fields[3], fields[7], order++});
    }
    std::stable_sort(events.begin(), events.end(), [](const Event &a,
                                                       const Event &b) {
        return a.arrival_ns < b.arrival_ns ||
            (a.arrival_ns == b.arrival_ns && a.input_order < b.input_order);
    });
    return events;
}

bool load_trajectory(std::vector<MpcTrajectorySample_t> *points,
                     double *lap_length_m)
{
    std::ifstream input(MPC_APPLIED_REPLAY_TRAJECTORY_PATH);
    if (!input.good()) return false;
    std::string line;
    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        if (fields.size() < 6) continue;
        std::array<double, 9> values{};
        const std::size_t count = std::min(fields.size(), values.size());
        bool valid = true;
        for (std::size_t i = 0; i < count; ++i) {
            if (!parse_number(fields[i], &values[i])) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;
        MpcTrajectorySample_t point{};
        point.s = values[0];
        point.x = values[1];
        point.y = values[2];
        point.heading = values[3];
        point.curvature = values[4];
        point.speed = values[5];
        point.left_bound = count >= 9 ? values[7] : 1.0;
        point.right_bound = count >= 9 ? values[8] : 1.0;
        points->push_back(point);
        if (points->size() > kMaximumTrajectoryPoints) return false;
    }
    std::size_t point_count = points->size();
    if (!mpc_trajectory_prepare(points->data(), &point_count, lap_length_m))
        return false;
    points->resize(point_count);
    return true;
}

bool parse_odom(const Event &event, MpcOdomSample *sample)
{
    if (!sample || event.header_stamp_ns <= 0) return false;
    double x = 0.0, y = 0.0, yaw = 0.0, u = 0.0, v = 0.0, r = 0.0;
    if (!json_number(event.payload, "x_m", &x) ||
        !json_number(event.payload, "y_m", &y) ||
        !json_number(event.payload, "yaw_rad", &yaw) ||
        !json_number(event.payload, "speed_mps", &u) ||
        !json_number(event.payload, "lateral_speed_mps", &v) ||
        !json_number(event.payload, "yaw_rate_radps", &r)) return false;
    *sample = MpcOdomSample{event.header_stamp_ns, x, y, yaw, u, v, r};
    return true;
}

bool parse_map_pose(const Event &event, MpcMapPoseAnchor *anchor)
{
    if (!anchor || event.header_stamp_ns <= 0) return false;
    double x = 0.0, y = 0.0, yaw = 0.0;
    if (!json_number(event.payload, "x_m", &x) ||
        !json_number(event.payload, "y_m", &y) ||
        !json_number(event.payload, "yaw_rad", &yaw)) return false;
    *anchor = MpcMapPoseAnchor{event.header_stamp_ns, x, y, yaw};
    return true;
}

bool parse_command(const Event &event, MpcCommandHistoryEntry *command)
{
    if (!command || event.header_stamp_ns <= 0) return false;
    double speed = 0.0, steering = 0.0;
    if (!json_number(event.payload, "speed_mps", &speed) ||
        !json_number(event.payload, "steering_angle_rad", &steering))
        return false;
    *command = MpcCommandHistoryEntry{event.header_stamp_ns, steering, speed};
    return true;
}

std::vector<MpcSynchronizedState> build_legal_reference(
    const std::vector<Event> &events)
{
    std::vector<MpcOdomSample> odometry;
    std::vector<MpcMapPoseAnchor> poses;
    for (const auto &event : events) {
        if (event.topic == "/odom") {
            MpcOdomSample sample{};
            if (parse_odom(event, &sample)) odometry.push_back(sample);
        } else if (event.topic == "/current_map_pose") {
            MpcMapPoseAnchor anchor{};
            if (parse_map_pose(event, &anchor)) poses.push_back(anchor);
        }
    }
    std::sort(odometry.begin(), odometry.end(), [](const auto &a, const auto &b) {
        return a.stamp_ns < b.stamp_ns;
    });
    std::sort(poses.begin(), poses.end(), [](const auto &a, const auto &b) {
        return a.stamp_ns < b.stamp_ns;
    });

    MpcStateSynchronizer synchronizer;
    std::vector<MpcSynchronizedState> states;
    std::size_t pose_index = 0;
    for (const auto &odom : odometry) {
        while (pose_index < poses.size() &&
               poses[pose_index].stamp_ns <= odom.stamp_ns) {
            if (synchronizer.set_map_pose(poses[pose_index]) != MpcSyncStatus::kOk)
                fail("future-reference map-pose ordering fault");
            ++pose_index;
        }
        if (synchronizer.push_odometry(odom) != MpcSyncStatus::kOk) continue;
        MpcSynchronizedState state{};
        if (synchronizer.synchronize(odom.stamp_ns, &state) == MpcSyncStatus::kOk)
            states.push_back(state);
    }
    return states;
}

bool build_continuous_projections(
    const std::vector<MpcSynchronizedState> &states,
    const std::vector<MpcTrajectorySample_t> &trajectory,
    double lap_length_m,
    std::vector<MpcPathProjection_t> *projections,
    std::vector<double> *unwrapped_progress)
{
    if (!projections || !unwrapped_progress || states.empty()) return false;
    projections->clear();
    unwrapped_progress->clear();
    projections->reserve(states.size());
    unwrapped_progress->reserve(states.size());
    std::size_t previous_segment = std::numeric_limits<std::size_t>::max();
    double progress = 0.0;
    for (const auto &state : states) {
        MpcPathProjection_t projection{};
        if (!mpc_trajectory_project(trajectory.data(), trajectory.size(),
                lap_length_m, state.map_x, state.map_y, state.map_yaw,
                previous_segment, 64, &projection)) return false;
        if (projections->empty()) {
            progress = projection.s;
        } else {
            double delta = projection.s - projections->back().s;
            if (delta > 0.5 * lap_length_m) delta -= lap_length_m;
            if (delta < -0.5 * lap_length_m) delta += lap_length_m;
            progress += delta;
        }
        projections->push_back(projection);
        unwrapped_progress->push_back(progress);
        previous_segment = projection.segment;
    }
    return true;
}

bool interpolate_reference(const std::vector<MpcSynchronizedState> &states,
                            const std::vector<MpcPathProjection_t> &projections,
                            const std::vector<double> &unwrapped_progress,
                            int64_t stamp_ns, double lap_length_m,
                            MpcSynchronizedState *result,
                            MpcPathProjection_t *projection,
                            double *progress_m)
{
    if (!result || !projection || !progress_m || states.size() < 2 ||
        projections.size() != states.size() ||
        unwrapped_progress.size() != states.size() ||
        stamp_ns < states.front().source_stamp_ns ||
        stamp_ns > states.back().source_stamp_ns) return false;
    const auto upper = std::lower_bound(states.begin(), states.end(), stamp_ns,
        [](const MpcSynchronizedState &state, int64_t stamp) {
            return state.source_stamp_ns < stamp;
        });
    if (upper == states.end()) return false;
    if (upper->source_stamp_ns == stamp_ns || upper == states.begin()) {
        *result = *upper;
        const std::size_t index = static_cast<std::size_t>(
            std::distance(states.begin(), upper));
        *projection = projections[index];
        *progress_m = unwrapped_progress[index];
        return true;
    }
    const auto lower = std::prev(upper);
    const std::size_t lower_index = static_cast<std::size_t>(
        std::distance(states.begin(), lower));
    const std::size_t upper_index = lower_index + 1;
    const double alpha = static_cast<double>(stamp_ns - lower->source_stamp_ns) /
        static_cast<double>(upper->source_stamp_ns - lower->source_stamp_ns);
    *result = *lower;
    result->source_stamp_ns = stamp_ns;
    result->map_x += alpha * (upper->map_x - lower->map_x);
    result->map_y += alpha * (upper->map_y - lower->map_y);
    result->map_yaw = std::atan2(
        std::sin(lower->map_yaw + alpha * std::atan2(
            std::sin(upper->map_yaw - lower->map_yaw),
            std::cos(upper->map_yaw - lower->map_yaw))),
        std::cos(lower->map_yaw + alpha * std::atan2(
            std::sin(upper->map_yaw - lower->map_yaw),
            std::cos(upper->map_yaw - lower->map_yaw))));
    result->u += alpha * (upper->u - lower->u);
    result->v += alpha * (upper->v - lower->v);
    result->yaw_rate += alpha * (upper->yaw_rate - lower->yaw_rate);
    result->source_age_s = 0.0;
    *progress_m = unwrapped_progress[lower_index] + alpha *
        (unwrapped_progress[upper_index] - unwrapped_progress[lower_index]);
    *projection = projections[lower_index];
    projection->s = std::fmod(*progress_m, lap_length_m);
    if (projection->s < 0.0) projection->s += lap_length_m;
    projection->lateral_error += alpha *
        (projections[upper_index].lateral_error - projection->lateral_error);
    return true;
}

double wrap_angle(double value)
{
    return std::atan2(std::sin(value), std::cos(value));
}

std::size_t speed_bin(double speed)
{
    if (speed < 0.5) return 0;
    if (speed < 2.0) return 1;
    if (speed < 4.0) return 2;
    if (speed < 6.0) return 3;
    if (speed < 8.0) return 4;
    return 5;
}

std::size_t command_at_or_before(
    const std::vector<MpcCommandHistoryEntry> &commands, int64_t stamp_ns)
{
    const auto upper = std::upper_bound(commands.begin(), commands.end(), stamp_ns,
        [](int64_t stamp, const MpcCommandHistoryEntry &command) {
            return stamp < command.stamp_ns;
        });
    return upper == commands.begin() ? commands.size() :
        static_cast<std::size_t>(std::distance(commands.begin(), upper) - 1);
}

bool rollout_actual_commands(
    const MpcSynchronizedState &source,
    const MpcPathProjection_t &source_projection,
    const std::vector<MpcCommandHistoryEntry> &commands,
    std::size_t initial_command_index,
    int horizon_steps,
    const std::vector<MpcTrajectorySample_t> &trajectory,
    double lap_length_m,
    MpcControlTimePrediction *prediction)
{
    if (!prediction || initial_command_index >= commands.size() ||
        horizon_steps <= 0) return false;

    MpcModelState_t plant{};
    plant.e_y = static_cast<float>(source_projection.lateral_error);
    plant.e_psi = static_cast<float>(source_projection.heading_error);
    plant.u = static_cast<float>(std::max(0.0, source.u));
    plant.v = static_cast<float>(source.v);
    plant.r = static_cast<float>(source.yaw_rate);
    plant.target_speed = static_cast<float>(commands[initial_command_index].target_speed_mps);
    plant.steering_command = static_cast<float>(
        commands[initial_command_index].steering_command_rad);
    double progress_m = source_projection.s;
    const int64_t target_stamp_ns = source.source_stamp_ns +
        static_cast<int64_t>(horizon_steps) * kControlStepNs;
    std::size_t next_command_index = initial_command_index + 1;
    int64_t cursor_ns = source.source_stamp_ns;

    while (cursor_ns < target_stamp_ns) {
        if (next_command_index < commands.size() &&
            commands[next_command_index].stamp_ns <= cursor_ns) {
            plant.target_speed = static_cast<float>(
                commands[next_command_index].target_speed_mps);
            plant.steering_command = static_cast<float>(
                commands[next_command_index].steering_command_rad);
            ++next_command_index;
            continue;
        }
        const int64_t next_change_ns = next_command_index < commands.size()
            ? commands[next_command_index].stamp_ns : target_stamp_ns;
        const int64_t segment_end_ns = std::min({target_stamp_ns,
            next_change_ns, cursor_ns + kControlStepNs});
        const float dt = static_cast<float>(
            static_cast<double>(segment_end_ns - cursor_ns) * 1.0e-9);
        MpcTrajectorySample_t path_sample{};
        if (!(dt > 0.0f) || !mpc_trajectory_sample(trajectory.data(),
                trajectory.size(), lap_length_m, progress_m, &path_sample))
            return false;
        const MpcModelControl_t held_command{0.0f, 0.0f};
        const MpcStageResult_t step = mpc_vehicle_model_step(&plant,
            &held_command, dt, static_cast<float>(path_sample.curvature));
        if (!step.valid) return false;
        plant = step.next;
        progress_m += step.delta_s_m;
        cursor_ns = segment_end_ns;
    }

    MpcTrajectorySample_t final_path_sample{};
    if (!mpc_trajectory_sample(trajectory.data(), trajectory.size(), lap_length_m,
            progress_m, &final_path_sample)) return false;
    MpcControlTimePrediction result{};
    result.state = source;
    result.state.source_stamp_ns = target_stamp_ns;
    result.state.map_x = final_path_sample.x -
        std::sin(final_path_sample.heading) * plant.e_y;
    result.state.map_y = final_path_sample.y +
        std::cos(final_path_sample.heading) * plant.e_y;
    result.state.map_yaw = wrap_angle(final_path_sample.heading + plant.e_psi);
    result.state.u = plant.u;
    result.state.v = plant.v;
    result.state.yaw_rate = plant.r;
    result.progress_m = progress_m;
    result.projection.s = std::fmod(progress_m, lap_length_m);
    if (result.projection.s < 0.0) result.projection.s += lap_length_m;
    result.projection.lateral_error = plant.e_y;
    result.projection.heading_error = plant.e_psi;
    result.projection.segment = source_projection.segment;
    *prediction = result;
    return true;
}

bool score_state(const MpcControlTimePrediction &predicted,
                 const MpcSynchronizedState &reference,
                 const MpcPathProjection_t &reference_projection,
                 double reference_progress_m, double lap_length_m,
                 std::array<double, 7> *error)
{
    if (!error) return false;
    double along_error = predicted.progress_m - reference_progress_m;
    along_error = std::fmod(along_error, lap_length_m);
    if (along_error > 0.5 * lap_length_m) along_error -= lap_length_m;
    if (along_error < -0.5 * lap_length_m) along_error += lap_length_m;
    (*error)[0] = std::hypot(predicted.state.map_x - reference.map_x,
                             predicted.state.map_y - reference.map_y);
    (*error)[1] = along_error;
    (*error)[2] = predicted.projection.lateral_error -
        reference_projection.lateral_error;
    (*error)[3] = wrap_angle(predicted.state.map_yaw - reference.map_yaw);
    (*error)[4] = predicted.state.u - reference.u;
    (*error)[5] = predicted.state.v - reference.v;
    (*error)[6] = predicted.state.yaw_rate - reference.yaw_rate;
    return true;
}

double percentile(const std::vector<double> &values, double fraction)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    if (fraction <= 0.0) return sorted.front();
    if (fraction >= 1.0) return sorted.back();
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted.size())) - 1.0);
    return sorted[std::min(index, sorted.size() - 1)];
}

void write_metrics(const std::string &path,
                   const std::array<std::array<std::array<Bucket, 7>, 6>, 5>
                       &buckets)
{
    std::ofstream output(path);
    if (!output.good()) fail("cannot create metrics CSV " + path);
    output << "horizon_steps,horizon_ms,initial_speed_bin,dimension,count,"
              "signed_mean,mae,p50_abs,p90_abs,p95_abs,p99_abs,max_abs\n";
    output << std::setprecision(10);
    for (std::size_t horizon = 0; horizon < buckets.size(); ++horizon) {
        for (std::size_t speed = 0; speed < buckets[horizon].size(); ++speed) {
            for (std::size_t dimension = 0; dimension < 7; ++dimension) {
                const auto &errors = buckets[horizon][speed][dimension].errors;
                double signed_sum = 0.0;
                double absolute_sum = 0.0;
                std::vector<double> absolute;
                absolute.reserve(errors.size());
                for (const double value : errors) {
                    signed_sum += value;
                    absolute_sum += std::abs(value);
                    absolute.push_back(std::abs(value));
                }
                const double count = static_cast<double>(errors.size());
                output << kHorizonSteps[horizon] << ','
            << kHorizonSteps[horizon] *
                static_cast<int>(kControlStepNs / 1'000'000LL) << ','
                    << kSpeedBinNames[speed] << ',' << kDimensionNames[dimension]
                    << ',' << errors.size() << ','
                    << (count ? signed_sum / count :
                        std::numeric_limits<double>::quiet_NaN()) << ','
                    << (count ? absolute_sum / count :
                        std::numeric_limits<double>::quiet_NaN()) << ','
                    << percentile(absolute, 0.50) << ','
                    << percentile(absolute, 0.90) << ','
                    << percentile(absolute, 0.95) << ','
                    << percentile(absolute, 0.99) << ','
                    << percentile(absolute, 1.00) << '\n';
            }
        }
    }
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "usage: mpc_applied_command_replay EVENTS_CSV OUTPUT_CSV\n";
        return 2;
    }
    std::vector<MpcTrajectorySample_t> trajectory;
    double lap_length_m = 0.0;
    if (!load_trajectory(&trajectory, &lap_length_m))
        fail("failed to load accepted raceline");
    const auto events = load_events(argv[1]);
    const auto legal_states = build_legal_reference(events);
    if (legal_states.size() < 2) fail("not enough legal state samples");
    std::vector<MpcPathProjection_t> legal_projections;
    std::vector<double> legal_progress;
    if (!build_continuous_projections(legal_states, trajectory, lap_length_m,
            &legal_projections, &legal_progress))
        fail("could not build continuous legal raceline projection");

    std::vector<MpcCommandHistoryEntry> commands;
    std::size_t invalid_or_future_commands = 0;
    for (const auto &event : events) {
        if (event.topic != "/cmd/speed") continue;
        MpcCommandHistoryEntry command{};
        if (!parse_command(event, &command) || command.stamp_ns > event.arrival_ns ||
            (!commands.empty() && command.stamp_ns <= commands.back().stamp_ns)) {
            ++invalid_or_future_commands;
            continue;
        }
        commands.push_back(command);
    }
    if (commands.empty()) fail("no causal /cmd/speed inputs were recorded");

    std::array<std::array<std::array<Bucket, 7>, 6>, 5> buckets{};
    std::array<std::size_t, 5> origins{};
    std::array<std::size_t, 5> scored{};
    std::size_t projection_failures = 0;
    std::size_t rollout_failures = 0;
    std::size_t missing_commands = 0;
    std::size_t reference_gaps = 0;

    for (std::size_t source_index = 0; source_index < legal_states.size();
         ++source_index) {
        const auto &source = legal_states[source_index];
        const std::size_t initial_command = command_at_or_before(commands,
            source.source_stamp_ns);
        if (initial_command >= commands.size()) {
            ++missing_commands;
            continue;
        }
        const MpcPathProjection_t &source_projection =
            legal_projections[source_index];
        const std::size_t speed = speed_bin(std::max(0.0, source.u));

        for (std::size_t horizon = 0; horizon < kHorizonSteps.size(); ++horizon) {
            const int steps = kHorizonSteps[horizon];
            const int64_t target_stamp_ns = source.source_stamp_ns +
                static_cast<int64_t>(steps) * kControlStepNs;
            MpcSynchronizedState reference{};
            MpcPathProjection_t reference_projection{};
            double reference_progress_m = 0.0;
            if (!interpolate_reference(legal_states, legal_projections,
                    legal_progress, target_stamp_ns, lap_length_m, &reference,
                    &reference_projection, &reference_progress_m)) {
                ++reference_gaps;
                continue;
            }
            ++origins[horizon];
            MpcControlTimePrediction predicted{};
            if (!rollout_actual_commands(source, source_projection, commands,
                    initial_command, steps, trajectory, lap_length_m, &predicted)) {
                ++rollout_failures;
                continue;
            }
            std::array<double, 7> errors{};
            if (!score_state(predicted, reference, reference_projection,
                    reference_progress_m, lap_length_m, &errors)) {
                ++projection_failures;
                continue;
            }
            ++scored[horizon];
            for (std::size_t dimension = 0; dimension < errors.size(); ++dimension)
                buckets[horizon][speed][dimension].errors.push_back(errors[dimension]);
        }
    }

    write_metrics(argv[2], buckets);
    std::cout << "events=" << argv[1] << '\n'
              << "accepted_model=mpc_vehicle_model_step; actual_inputs=/cmd/speed; "
                 "held targets and steering are applied at their recorded header stamps\n"
              << "reference=interpolated later legal /odom + /current_map_pose; "
                 "simulator_time_and_truth_not_read\n"
              << "command_samples=" << commands.size()
              << " invalid_or_future_commands=" << invalid_or_future_commands << '\n'
              << "legal_state_samples=" << legal_states.size()
              << " missing_initial_commands=" << missing_commands << '\n'
              << "projection_failures=" << projection_failures
              << " rollout_failures=" << rollout_failures
              << " reference_gaps=" << reference_gaps << '\n';
    for (std::size_t horizon = 0; horizon < kHorizonSteps.size(); ++horizon)
        std::cout << "N" << kHorizonSteps[horizon] << "_origins="
                  << origins[horizon] << " scored=" << scored[horizon] << '\n';
    std::cout << "metrics_csv=" << argv[2] << '\n';
    return 0;
}

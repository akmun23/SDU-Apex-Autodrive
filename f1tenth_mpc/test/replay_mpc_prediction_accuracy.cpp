#define _POSIX_C_SOURCE 200809L

#include "mpc_control_time_predictor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef MPC_PREDICTION_REPLAY_TRAJECTORY_PATH
#error "MPC_PREDICTION_REPLAY_TRAJECTORY_PATH must name the accepted raceline"
#endif

using namespace f1tenth_mpc;

namespace {

constexpr std::size_t kMaximumTrajectoryPoints = 4000;
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<double, 7> kAgeBinEdgesMs{{0.0, 25.0, 40.0, 60.0,
                                                80.0, 100.0, 120.0}};
constexpr std::array<const char *, 6> kAgeBinNames{{"0-25", "25-40", "40-60",
                                                    "60-80", "80-100", "100-120"}};
constexpr std::array<const char *, 3> kModeNames{{"CT0", "CT1", "CT2"}};
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

struct LegalOdom {
    MpcOdomSample sample{};
};

struct LegalPose {
    MpcMapPoseAnchor anchor{};
};

struct StateError {
    std::array<double, 7> value{};
};

struct Bucket {
    std::array<std::vector<double>, 7> errors;
    std::size_t evaluated{};
};

void fail(const std::string &message)
{
    std::cerr << "prediction replay error: " << message << '\n';
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

bool load_trajectory(std::vector<MpcTrajectorySample_t> *points,
                     double *lap_length_m)
{
    std::ifstream input(MPC_PREDICTION_REPLAY_TRAJECTORY_PATH);
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
        point.acceleration = count >= 7 ? values[6] : 0.0;
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

double wrap_angle(double value)
{
    return std::atan2(std::sin(value), std::cos(value));
}

bool interpolate_reference(const std::vector<MpcSynchronizedState> &states,
                            int64_t stamp_ns, MpcSynchronizedState *result)
{
    if (!result || states.size() < 2 || stamp_ns < states.front().source_stamp_ns ||
        stamp_ns > states.back().source_stamp_ns) return false;
    const auto upper = std::lower_bound(states.begin(), states.end(), stamp_ns,
        [](const MpcSynchronizedState &state, int64_t stamp) {
            return state.source_stamp_ns < stamp;
        });
    if (upper == states.end()) return false;
    if (upper->source_stamp_ns == stamp_ns || upper == states.begin()) {
        *result = *upper;
        return true;
    }
    const auto lower = std::prev(upper);
    const double alpha = static_cast<double>(stamp_ns - lower->source_stamp_ns) /
        static_cast<double>(upper->source_stamp_ns - lower->source_stamp_ns);
    *result = *lower;
    result->source_stamp_ns = stamp_ns;
    result->map_x += alpha * (upper->map_x - lower->map_x);
    result->map_y += alpha * (upper->map_y - lower->map_y);
    result->map_yaw = wrap_angle(lower->map_yaw + alpha *
        wrap_angle(upper->map_yaw - lower->map_yaw));
    result->u += alpha * (upper->u - lower->u);
    result->v += alpha * (upper->v - lower->v);
    result->yaw_rate += alpha * (upper->yaw_rate - lower->yaw_rate);
    result->source_age_s = 0.0;
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
        if (synchronizer.push_odometry(odom) != MpcSyncStatus::kOk)
            continue;
        MpcSynchronizedState state{};
        if (synchronizer.synchronize(odom.stamp_ns, &state) == MpcSyncStatus::kOk)
            states.push_back(state);
    }
    return states;
}

bool signed_along_error(double first_s, double second_s, double lap_length,
                        double *error)
{
    if (!error || !(lap_length > 0.0)) return false;
    double difference = std::fmod(first_s - second_s, lap_length);
    if (difference > 0.5 * lap_length) difference -= lap_length;
    if (difference < -0.5 * lap_length) difference += lap_length;
    *error = difference;
    return true;
}

bool score_state(const MpcControlTimePrediction &predicted,
                 const MpcSynchronizedState &reference,
                 const std::vector<MpcTrajectorySample_t> &trajectory,
                 double lap_length, StateError *error)
{
    MpcPathProjection_t reference_projection{};
    if (!mpc_trajectory_project(trajectory.data(), trajectory.size(), lap_length,
            reference.map_x, reference.map_y, reference.map_yaw,
            std::numeric_limits<std::size_t>::max(), 0,
            &reference_projection)) return false;
    if (!signed_along_error(predicted.projection.s, reference_projection.s,
            lap_length, &error->value[1])) return false;
    error->value[0] = std::hypot(predicted.state.map_x - reference.map_x,
                                 predicted.state.map_y - reference.map_y);
    error->value[2] = predicted.projection.lateral_error -
        reference_projection.lateral_error;
    error->value[3] = wrap_angle(predicted.state.map_yaw - reference.map_yaw);
    error->value[4] = predicted.state.u - reference.u;
    error->value[5] = predicted.state.v - reference.v;
    error->value[6] = predicted.state.yaw_rate - reference.yaw_rate;
    return true;
}

std::size_t age_bin(double age_ms)
{
    for (std::size_t i = 0; i + 1 < kAgeBinEdgesMs.size(); ++i) {
        if (age_ms >= kAgeBinEdgesMs[i] && age_ms < kAgeBinEdgesMs[i + 1])
            return i;
    }
    return age_ms == kAgeBinEdgesMs.back() ? kAgeBinNames.size() - 1 :
        kAgeBinNames.size();
}

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    if (fraction <= 0.0) return values.front();
    if (fraction >= 1.0) return values.back();
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

void write_metrics(const std::string &path,
                   const std::array<std::array<Bucket, 6>, 3> &buckets)
{
    std::ofstream output(path);
    if (!output.good()) fail("cannot create prediction metric CSV " + path);
    output << "mode,age_bin_ms,dimension,count,signed_mean,mae,p50_abs,p90_abs,"
              "p95_abs,p99_abs,max_abs\n";
    output << std::setprecision(10);
    for (std::size_t mode = 0; mode < buckets.size(); ++mode) {
        for (std::size_t bin = 0; bin < buckets[mode].size(); ++bin) {
            const auto &bucket = buckets[mode][bin];
            for (std::size_t dimension = 0; dimension < 7; ++dimension) {
                const auto &errors = bucket.errors[dimension];
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
                output << kModeNames[mode] << ',' << kAgeBinNames[bin] << ','
                    << kDimensionNames[dimension] << ',' << errors.size() << ','
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
        std::cerr << "usage: mpc_prediction_accuracy_replay EVENTS_CSV OUTPUT_CSV\n";
        return 2;
    }
    const std::string event_path = argv[1];
    const std::string output_path = argv[2];
    std::vector<MpcTrajectorySample_t> trajectory;
    double lap_length = 0.0;
    if (!load_trajectory(&trajectory, &lap_length))
        fail("failed to load accepted raceline");
    const auto events = load_events(event_path);
    const auto future_reference = build_legal_reference(events);
    if (future_reference.size() < 2)
        fail("not enough later legal /odom and /current_map_pose samples");

    std::array<std::array<Bucket, 6>, 3> buckets{};
    std::array<std::vector<double>, 3> predictor_time_us;
    MpcStateSynchronizer synchronizer;
    MpcCommandHistory command_history;
    MpcControlTimePredictorConfig predictor_config;
    std::size_t odom_events = 0;
    std::size_t synchronized = 0;
    std::size_t reference_available = 0;
    std::size_t prediction_failures = 0;
    std::size_t rejected_commands = 0;
    std::size_t command_history_size_high_water = 0;
    std::size_t previous_segment = std::numeric_limits<std::size_t>::max();

    for (const auto &event : events) {
        if (event.topic == "/current_map_pose") {
            MpcMapPoseAnchor anchor{};
            if (parse_map_pose(event, &anchor))
                synchronizer.set_map_pose(anchor);
        } else if (event.topic == "/cmd/speed") {
            MpcCommandHistoryEntry command{};
            if (!parse_command(event, &command) ||
                command.stamp_ns > event.arrival_ns ||
                !command_history.push(command)) ++rejected_commands;
            command_history_size_high_water = std::max(
                command_history_size_high_water, command_history.size());
        } else if (event.topic == "/odom") {
            ++odom_events;
            MpcOdomSample odom{};
            if (!parse_odom(event, &odom) ||
                synchronizer.push_odometry(odom) != MpcSyncStatus::kOk)
                continue;
            MpcSynchronizedState source{};
            if (synchronizer.synchronize(event.arrival_ns, &source) !=
                MpcSyncStatus::kOk) continue;
            ++synchronized;

            MpcControlTimePrediction predictions[3]{};
            const std::array<MpcControlTimeMode, 3> modes{{
                MpcControlTimeMode::kNoExtrapolation,
                MpcControlTimeMode::kConstantBodyTwist,
                MpcControlTimeMode::kAcceptedModelCommandHistory}};
            bool valid_all = true;
            for (std::size_t mode = 0; mode < modes.size(); ++mode) {
                const auto prediction_start = std::chrono::steady_clock::now();
                const auto status = predict_to_control_time(modes[mode], source,
                    event.arrival_ns, command_history, trajectory.data(),
                    trajectory.size(), lap_length, previous_segment, nullptr,
                    predictor_config, &predictions[mode]);
                const auto prediction_finish = std::chrono::steady_clock::now();
                predictor_time_us[mode].push_back(
                    std::chrono::duration<double, std::micro>(
                        prediction_finish - prediction_start).count());
                if (status != MpcControlTimeStatus::kOk) {
                    ++prediction_failures;
                    valid_all = false;
                    break;
                }
            }
            if (!valid_all) continue;
            previous_segment = predictions[0].projection.segment;
            MpcSynchronizedState reference{};
            if (!interpolate_reference(future_reference, event.arrival_ns,
                    &reference)) continue;
            ++reference_available;
            const double age_ms = predictions[0].age_s * 1000.0;
            const std::size_t bin = age_bin(age_ms);
            if (bin >= kAgeBinNames.size()) continue;
            for (std::size_t mode = 0; mode < 3; ++mode) {
                StateError errors{};
                if (!score_state(predictions[mode], reference, trajectory,
                        lap_length, &errors)) continue;
                ++buckets[mode][bin].evaluated;
                for (std::size_t dimension = 0; dimension < errors.value.size();
                     ++dimension)
                    buckets[mode][bin].errors[dimension].push_back(
                        errors.value[dimension]);
            }
        }
    }

    /* The replay always scores all three predeclared variants. */
    if (reference_available == 0)
        fail("no predictor outputs could be scored at a later legal timestamp");
    write_metrics(output_path, buckets);
    std::cout << "events=" << event_path << '\n'
              << "future_legal_reference_samples=" << future_reference.size() << '\n'
              << "odom_events=" << odom_events << " synchronized=" << synchronized
              << " scored_targets=" << reference_available << '\n'
              << "prediction_failures=" << prediction_failures
              << " rejected_or_noncausal_commands=" << rejected_commands << '\n'
              << "command_history_capacity=" << kMpcCommandHistoryCapacity
              << " high_water=" << command_history_size_high_water << '\n'
              << "scoring_reference=interpolated future /odom + /current_map_pose; "
                 "offline only\n"
              << "predictor_inputs=events received by callback; simulation_time_s "
                 "and simulator-only topics are not read\n"
              << "metrics_csv=" << output_path << '\n';
    for (std::size_t mode = 0; mode < kModeNames.size(); ++mode) {
        std::cout << kModeNames[mode] << "_predictor_time_us[p50,p95,p99,max]="
            << percentile(predictor_time_us[mode], 0.50) << ','
            << percentile(predictor_time_us[mode], 0.95) << ','
            << percentile(predictor_time_us[mode], 0.99) << ','
            << percentile(predictor_time_us[mode], 1.00) << '\n';
    }
    return 0;
}

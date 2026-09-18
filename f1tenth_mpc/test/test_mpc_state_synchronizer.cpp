#include "mpc_state_synchronizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifndef MPC_SYNC_TEST_EVENTS_PATH
#error "MPC_SYNC_TEST_EVENTS_PATH must point at the recorded event stream"
#endif

using namespace f1tenth_mpc;

static int failures = 0;

static void check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

static void check_near(double actual, double expected, double tolerance,
                       const char *message)
{
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message);
}

static MpcOdomSample odom(int64_t stamp, double x, double y, double yaw,
                          double u = 0.0, double v = 0.0,
                          double r = 0.0)
{
    MpcOdomSample sample;
    sample.stamp_ns = stamp;
    sample.x = x;
    sample.y = y;
    sample.yaw = yaw;
    sample.u = u;
    sample.v = v;
    sample.yaw_rate = r;
    return sample;
}

static void test_se2_anchor_catchup_and_interpolation()
{
    MpcStateSynchronizer sync;
    check(sync.push_odometry(odom(1000000000, 1.0, 1.0, 1.57079632679)) ==
          MpcSyncStatus::kOk, "accept first ordered odometry sample");
    check(sync.push_odometry(odom(1050000000, 1.0, 2.0, 1.57079632679)) ==
          MpcSyncStatus::kOk, "accept second odometry sample");
    check(sync.push_odometry(odom(1100000000, 1.0, 3.0, 1.57079632679,
                                  3.0, 0.1, 0.2)) == MpcSyncStatus::kOk,
          "accept newest odometry sample");
    check(sync.set_map_pose({1050000000, 10.0, 20.0, 1.57079632679}) ==
          MpcSyncStatus::kOk, "accept global map-pose anchor");

    MpcSynchronizedState state;
    check(sync.synchronize(1150000000, &state) == MpcSyncStatus::kOk,
          "align map pose to latest odometry and command time");
    check_near(state.map_x, 10.0, 1.0e-8,
               "SE2 catch-up rotates relative x into map frame");
    check_near(state.map_y, 21.0, 1.0e-8,
               "SE2 catch-up propagates odometry delta from map anchor");
    check_near(state.map_yaw, 1.57079632679, 1.0e-8,
               "SE2 catch-up preserves coherent global yaw");
    check_near(state.u, 3.0, 1.0e-12,
               "synchronized state carries newest legal odometry velocity");
    check_near(state.source_age_s, 0.05, 1.0e-12,
               "state age is measured from source time to command time");
    check(state.odom_source_stamp_ns == 1100000000LL,
          "retain the odometry source timestamp used for synchronization");
    check(state.map_pose_source_stamp_ns == 1050000000LL,
          "retain the independent map-pose source timestamp");
    check_near(state.pose_odom_skew_s, 0.05, 1.0e-12,
               "pose-to-odometry skew is reported separately");
}

static void test_timing_jitter_is_accepted_and_reset_clears_state()
{
    MpcStateSynchronizer sync;
    sync.push_odometry(odom(1000000000, 0.0, 0.0, 0.0));
    sync.push_odometry(odom(1050000000, 0.0, 0.0, 0.0));
    sync.set_map_pose({1050000000, 0.0, 0.0, 0.0});
    MpcSynchronizedState state;

    check(sync.synchronize(1040000000, &state) == MpcSyncStatus::kOk,
          "accept a command timestamp earlier than the newest received state");
    check_near(state.source_age_s, -0.01, 1.0e-12,
               "retain signed timestamp offset as a diagnostic");
    check(sync.synchronize(1180000000, &state) == MpcSyncStatus::kOk,
          "accept source state beyond configured age support");
    check_near(state.source_age_s, 0.13, 1.0e-12,
               "retain measured source age without a timing rejection");

    sync.reset();
    check(sync.synchronize(1100000000, &state) == MpcSyncStatus::kMissingOdom,
          "reset clears stale source samples and epoch state");
    sync.push_odometry(odom(2000000000, 0.0, 0.0, 0.0));
    sync.push_odometry(odom(2050000000, 0.0, 0.0, 0.0));
    sync.set_map_pose({2100000000, 0.0, 0.0, 0.0});
    check(sync.synchronize(2150000000, &state) == MpcSyncStatus::kOk,
          "accept a map anchor newer than the newest odometry sample");
    check_near(state.map_x, 0.0, 1.0e-12,
               "future map anchor remains the best available position");

    sync.reset();
    sync.push_odometry(odom(3000000000, 0.0, 0.0, 0.0));
    check(sync.push_odometry(odom(2990000000, 0.0, 0.0, 0.0)) ==
          MpcSyncStatus::kOk,
          "timestamp reversal rebases the window without latching a fault");
    sync.reset();
    check(sync.push_odometry(odom(4000000000, 0.0, 0.0, 0.0)) ==
          MpcSyncStatus::kOk,
          "synchronizer continues accepting source samples");
}

static std::vector<std::string> split_csv(const std::string &line)
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

static bool json_number(const std::string &json, const char *key, double *value)
{
    const std::string token = std::string("\"") + key + "\"";
    const std::size_t key_position = json.find(token);
    if (key_position == std::string::npos) return false;
    const std::size_t colon = json.find(':', key_position + token.size());
    if (colon == std::string::npos) return false;
    char *end = nullptr;
    const double parsed = std::strtod(json.c_str() + colon + 1, &end);
    if (end == json.c_str() + colon + 1 || !std::isfinite(parsed)) return false;
    *value = parsed;
    return true;
}

static void test_failed_run_replay_uses_only_legal_runtime_topics()
{
    std::ifstream input(MPC_SYNC_TEST_EVENTS_PATH);
    check(input.good(), "open failed 4 m/s event trace for legal-input replay");
    if (!input.good()) return;

    MpcSyncConfig config;
    config.max_pose_odom_skew_s = 0.12;
    config.max_state_age_s = 0.12;
    MpcStateSynchronizer sync(config);
    std::string line;
    std::vector<double> accepted_ages;
    std::size_t odom_count = 0;
    std::size_t accepted_count = 0;
    std::size_t order_faults = 0;
    std::size_t source_gap_faults = 0;
    std::size_t other_rejections = 0;

    while (std::getline(input, line)) {
        const auto fields = split_csv(line);
        if (fields.size() < 8 || fields[0] == "event_index") continue;
        const std::string &topic = fields[3];
        if (topic != "/odom" && topic != "/current_map_pose") continue;
        const int64_t arrival_epoch_ns = std::stoll(fields[2]);
        const int64_t source_stamp_ns = std::stoll(fields[5]);
        const std::string &payload = fields[7];

        double x = 0.0, y = 0.0, yaw = 0.0;
        if (!json_number(payload, "x_m", &x) ||
            !json_number(payload, "y_m", &y) ||
            !json_number(payload, "yaw_rad", &yaw)) {
            ++other_rejections;
            continue;
        }

        if (topic == "/current_map_pose") {
            const auto result = sync.set_map_pose({source_stamp_ns, x, y, yaw});
            if (result == MpcSyncStatus::kTimestampOrderFault) ++order_faults;
            continue;
        }

        double u = 0.0, v = 0.0, r = 0.0;
        if (!json_number(payload, "speed_mps", &u) ||
            !json_number(payload, "lateral_speed_mps", &v) ||
            !json_number(payload, "yaw_rate_radps", &r)) {
            ++other_rejections;
            continue;
        }
        ++odom_count;
        const auto push_status = sync.push_odometry(
            {source_stamp_ns, x, y, yaw, u, v, r});
        if (push_status == MpcSyncStatus::kTimestampOrderFault) {
            ++order_faults;
            continue;
        }
        if (push_status == MpcSyncStatus::kSourceGapFault) {
            ++source_gap_faults;
            continue;
        }
        if (push_status != MpcSyncStatus::kOk) {
            ++other_rejections;
            continue;
        }

        MpcSynchronizedState state;
        const auto result = sync.synchronize(arrival_epoch_ns, &state);
        if (result == MpcSyncStatus::kOk) {
            ++accepted_count;
            accepted_ages.push_back(state.source_age_s);
        } else if (result == MpcSyncStatus::kTimestampOrderFault) {
            ++order_faults;
        } else if (result != MpcSyncStatus::kMissingMapPose &&
                   result != MpcSyncStatus::kNoOdomBracket) {
            ++other_rejections;
        }
    }

    check(odom_count >= 2000,
          "replay consumes the recorded consecutive legal odometry stream");
    check(order_faults == 0,
          "recorded odometry and map-pose source order remains monotonic");
    check(source_gap_faults == 0,
          "recorded source interval jitter never creates a gap rejection");
    check(accepted_count > odom_count * 9 / 10,
          "ordinary delayed states are synchronizable without the old 75 ms gate");
    if (!accepted_ages.empty()) {
        std::sort(accepted_ages.begin(), accepted_ages.end());
        const double p95 = accepted_ages[
            static_cast<std::size_t>(0.95 * (accepted_ages.size() - 1))];
        const double maximum = accepted_ages.back();
        std::cout << "4 m/s legal-state replay: accepted " << accepted_count << "/"
                  << odom_count << ", source age p95=" << p95 * 1000.0
                  << " ms max=" << maximum * 1000.0 << " ms; order faults="
                  << order_faults << " gap faults=" << source_gap_faults
                  << " other rejections=" << other_rejections << '\n';
        check(p95 <= 0.12,
              "replayed ordinary state-age distribution fits configured catch-up support");
    }
}

int main()
{
    test_se2_anchor_catchup_and_interpolation();
    test_timing_jitter_is_accepted_and_reset_clears_state();
    test_failed_run_replay_uses_only_legal_runtime_topics();
    if (failures != 0) {
        std::cerr << failures << " MPC state synchronizer test(s) failed\n";
        return 1;
    }
    std::cout << "MPC state synchronizer tests passed\n";
    return 0;
}

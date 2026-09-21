#include "mpc_state_synchronizer.hpp"

#include <cmath>
#include <iostream>

using namespace f1tenth_mpc;

namespace {

int failures = 0;

void check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_near(double actual, double expected, double tolerance,
                const char *message)
{
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message);
}

MpcOdomSample odom(int64_t stamp, double x, double y, double yaw,
                   double u = 0.0, double v = 0.0, double r = 0.0)
{
    return {stamp, x, y, yaw, u, v, r};
}

void test_se2_anchor_and_interpolation()
{
    MpcStateSynchronizer sync;
    check(sync.push_odometry(odom(1000000000, 1.0, 1.0, 1.57079632679)) ==
              MpcSyncStatus::kOk,
          "accept first ordered odometry sample");
    check(sync.push_odometry(odom(1050000000, 1.0, 2.0, 1.57079632679)) ==
              MpcSyncStatus::kOk,
          "accept second ordered odometry sample");
    check(sync.push_odometry(odom(1100000000, 1.0, 3.0, 1.57079632679,
                                 3.0, 0.1, 0.2)) == MpcSyncStatus::kOk,
          "accept newest ordered odometry sample");
    check(sync.set_map_pose({1050000000, 10.0, 20.0, 1.57079632679}) ==
              MpcSyncStatus::kOk,
          "accept global map-pose anchor");

    MpcSynchronizedState state;
    check(sync.synchronize(1150000000, &state) == MpcSyncStatus::kOk,
          "align map pose to latest odometry and command time");
    check_near(state.map_x, 10.0, 1.0e-8,
               "rotate relative x into the map frame");
    check_near(state.map_y, 21.0, 1.0e-8,
               "propagate odometry delta from map anchor");
    check_near(state.map_yaw, 1.57079632679, 1.0e-8,
               "preserve coherent global yaw");
    check_near(state.u, 3.0, 1.0e-12,
               "carry newest legal odometry velocity");
    check_near(state.source_age_s, 0.05, 1.0e-12,
               "measure source age from source to command time");
    check(state.odom_source_stamp_ns == 1100000000LL,
          "retain odometry source timestamp");
    check(state.map_pose_source_stamp_ns == 1050000000LL,
          "retain map-pose source timestamp");
}

void test_jitter_is_accepted_and_reset_clears_state()
{
    MpcStateSynchronizer sync;
    sync.push_odometry(odom(1000000000, 0.0, 0.0, 0.0));
    sync.push_odometry(odom(1050000000, 0.0, 0.0, 0.0));
    sync.set_map_pose({1050000000, 0.0, 0.0, 0.0});

    MpcSynchronizedState state;
    check(sync.synchronize(1040000000, &state) == MpcSyncStatus::kOk,
          "accept command time earlier than newest received state");
    check_near(state.source_age_s, -0.01, 1.0e-12,
               "retain signed timestamp offset as a diagnostic");
    check(sync.synchronize(1180000000, &state) == MpcSyncStatus::kOk,
          "accept delayed source state without timing rejection");
    check_near(state.source_age_s, 0.13, 1.0e-12,
               "retain measured delayed source age");

    sync.reset();
    check(sync.synchronize(1100000000, &state) == MpcSyncStatus::kMissingOdom,
          "reset clears stale source samples");
    sync.push_odometry(odom(2000000000, 0.0, 0.0, 0.0));
    sync.push_odometry(odom(2050000000, 0.0, 0.0, 0.0));
    sync.set_map_pose({2100000000, 0.0, 0.0, 0.0});
    check(sync.synchronize(2150000000, &state) == MpcSyncStatus::kOk,
          "accept a map anchor newer than odometry");
}

}  // namespace

int main()
{
    test_se2_anchor_and_interpolation();
    test_jitter_is_accepted_and_reset_clears_state();
    if (failures != 0) {
        std::cerr << failures << " MPC state synchronizer test(s) failed\n";
        return 1;
    }
    std::cout << "MPC state synchronizer tests passed\n";
    return 0;
}

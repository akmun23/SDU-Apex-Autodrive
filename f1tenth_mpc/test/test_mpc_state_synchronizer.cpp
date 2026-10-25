#include "mpc_state_synchronizer.hpp"

#include <gtest/gtest.h>

#include <limits>

using f1tenth_mpc::is_causal_plant_state_usable;
using f1tenth_mpc::MpcFreshnessDecision;
using f1tenth_mpc::MpcStateFreshnessGuard;
using f1tenth_mpc::MpcStateSynchronizer;
using f1tenth_mpc::MpcSyncConfig;
using f1tenth_mpc::MpcSyncStatus;

TEST(MpcStateFreshnessGuard, WaitsThroughTransientStalenessThenRecovers)
{
  MpcStateFreshnessGuard guard(0.5);
  constexpr int64_t start_ns = 7'000'000'000;
  EXPECT_EQ(guard.observe(MpcSyncStatus::kStateTooOld, start_ns),
            MpcFreshnessDecision::kWaitForFreshState);
  EXPECT_EQ(guard.observe(MpcSyncStatus::kStateTooOld,
                          start_ns + 300'000'000),
            MpcFreshnessDecision::kWaitForFreshState);
  EXPECT_NEAR(guard.stale_duration_s(start_ns + 300'000'000), 0.3, 1e-9);
  EXPECT_EQ(guard.observe(MpcSyncStatus::kOk, start_ns + 310'000'000),
            MpcFreshnessDecision::kReady);
  EXPECT_DOUBLE_EQ(guard.stale_duration_s(start_ns + 320'000'000), 0.0);
}

TEST(MpcStateFreshnessGuard, FaultsOnlyAfterGraceIsExceeded)
{
  MpcStateFreshnessGuard guard(0.5);
  constexpr int64_t start_ns = 8'000'000'000;
  EXPECT_EQ(guard.observe(MpcSyncStatus::kStateTooOld, start_ns),
            MpcFreshnessDecision::kWaitForFreshState);
  EXPECT_EQ(guard.observe(MpcSyncStatus::kStateTooOld,
                          start_ns + 500'000'000),
            MpcFreshnessDecision::kWaitForFreshState);
  EXPECT_EQ(guard.observe(MpcSyncStatus::kStateTooOld,
                          start_ns + 500'000'001),
            MpcFreshnessDecision::kFault);
}

TEST(MpcStateFreshnessGuard, DoesNotGraceOtherSynchronizationFaults)
{
  MpcStateFreshnessGuard guard(0.5);
  constexpr int64_t now_ns = 9'000'000'000;
  EXPECT_EQ(guard.observe(MpcSyncStatus::kPoseOdomSkewExceeded, now_ns),
            MpcFreshnessDecision::kFault);
  EXPECT_EQ(guard.observe(MpcSyncStatus::kStateTooOld, now_ns + 1),
            MpcFreshnessDecision::kWaitForFreshState);
}

TEST(MpcStateFreshnessGuard, ZeroGraceFaultsImmediately)
{
  MpcStateFreshnessGuard guard(0.0);
  EXPECT_EQ(guard.observe(MpcSyncStatus::kStateTooOld, 10'000'000'000),
            MpcFreshnessDecision::kFault);
}

TEST(MpcStateSynchronizer, ReturnsSourceTimingWhenRejectingStaleState)
{
  MpcSyncConfig config;
  config.max_state_age_s = 0.12;
  MpcStateSynchronizer synchronizer(config);
  constexpr int64_t source_ns = 10'000'000'000;
  EXPECT_EQ(synchronizer.push_odometry({source_ns, 1.0, 2.0, 0.3,
                                         4.0, 0.1, 0.2}),
            MpcSyncStatus::kOk);
  EXPECT_EQ(synchronizer.set_map_pose({source_ns, 5.0, 6.0, 0.7}),
            MpcSyncStatus::kOk);
  f1tenth_mpc::MpcSynchronizedState state{};
  EXPECT_EQ(synchronizer.synchronize(source_ns + 300'000'000, &state),
            MpcSyncStatus::kStateTooOld);
  EXPECT_EQ(state.source_stamp_ns, source_ns);
  EXPECT_EQ(state.odom_source_stamp_ns, source_ns);
  EXPECT_EQ(state.map_pose_source_stamp_ns, source_ns);
  EXPECT_NEAR(state.source_age_s, 0.3, 1e-9);
}

TEST(MpcPlantStateHandoff, AcceptsMatchingAndOnePacketDelayedSnapshots)
{
  constexpr int64_t odom_stamp = 10'000'000'000;
  EXPECT_TRUE(is_causal_plant_state_usable(
      true, odom_stamp, odom_stamp, odom_stamp, 0.120));
  EXPECT_TRUE(is_causal_plant_state_usable(
      true, odom_stamp - 25'000'000, odom_stamp, odom_stamp, 0.120));
}

TEST(MpcPlantStateHandoff, RejectsFutureOrStaleSnapshots)
{
  constexpr int64_t odom_stamp = 10'000'000'000;
  EXPECT_FALSE(is_causal_plant_state_usable(
      true, odom_stamp + 1, odom_stamp, odom_stamp + 1, 0.120));
  EXPECT_FALSE(is_causal_plant_state_usable(
      true, odom_stamp - 121'000'000, odom_stamp, odom_stamp, 0.120));
  EXPECT_FALSE(is_causal_plant_state_usable(
      true, odom_stamp - 1, odom_stamp,
      odom_stamp - 25'000'000, 0.120));
}

TEST(MpcPlantStateHandoff, RejectsMissingOrInvalidSnapshots)
{
  constexpr int64_t stamp = 10'000'000'000;
  EXPECT_FALSE(is_causal_plant_state_usable(false, stamp, stamp, stamp, 0.120));
  EXPECT_FALSE(is_causal_plant_state_usable(true, 0, stamp, stamp, 0.120));
  EXPECT_FALSE(is_causal_plant_state_usable(
      true, stamp, stamp, stamp,
      std::numeric_limits<double>::quiet_NaN()));
}

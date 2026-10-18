#include "mpc_state_synchronizer.hpp"

#include <gtest/gtest.h>

#include <limits>

using f1tenth_mpc::is_causal_plant_state_usable;

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

#include "f1tenth_localization/bridge_request_command_history.hpp"
#include <gtest/gtest.h>
#include <limits>

using f1tenth_localization::BridgeRequestCommandHistory;

TEST(BridgeRequestCommandHistory, ExactStampPreventsCallbackPhaseError)
{
  BridgeRequestCommandHistory history;
  BridgeRequestCommandHistory::Command selected{};
  constexpr int64_t request = 1793543092414731421LL;
  ASSERT_TRUE(history.push(request - 25000000LL, -.644, .079));
  ASSERT_TRUE(history.push(request, -.758, .079));
  // A newer command received just after the request is NOT used in the
  // preceding physical interval, even if it arrives before sensor callbacks.
  ASSERT_TRUE(history.push(request + 1000000LL, -.790, .121));
  ASSERT_TRUE(history.find_exact(request, &selected));
  EXPECT_EQ(selected.stamp_ns, request);
  EXPECT_DOUBLE_EQ(selected.steering_normalized, -.758);
  EXPECT_DOUBLE_EQ(selected.throttle_normalized, .079);
  EXPECT_FALSE(history.find_exact(request + 7LL, &selected));
}

TEST(BridgeRequestCommandHistory, RejectsInvalidAndOverwritesOldRecords)
{
  BridgeRequestCommandHistory history;
  BridgeRequestCommandHistory::Command selected{};
  EXPECT_FALSE(history.push(0, 0., .5));
  EXPECT_FALSE(history.push(1234, 0., std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(history.push(1234, 1.2, .5));
  EXPECT_FALSE(history.push(1234, 0., -1.2));
  EXPECT_FALSE(history.find_exact(1234, &selected));
  for (std::size_t i=0; i<BridgeRequestCommandHistory::kCapacity + 5; ++i) {
    ASSERT_TRUE(history.push(100000LL + static_cast<int64_t>(i), .2, .3));
  }
  EXPECT_FALSE(history.find_exact(100000LL, &selected));
  EXPECT_TRUE(history.find_exact(
      100000LL + BridgeRequestCommandHistory::kCapacity + 4, &selected));
  EXPECT_DOUBLE_EQ(selected.throttle_normalized, .3);
}

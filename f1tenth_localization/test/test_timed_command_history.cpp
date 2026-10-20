#include "f1tenth_localization/timed_command_history.hpp"

#include <gtest/gtest.h>

namespace f1tenth_localization
{
namespace
{

TEST(TimedCommandHistory, ExcludesCommandsPublishedAfterPacketEpoch)
{
  TimedCommandHistory history;
  history.push(100, -0.4);
  history.push(300, 0.7);

  EXPECT_DOUBLE_EQ(history.value_at_or_before(99, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(history.value_at_or_before(200, 0.0), -0.4);
  EXPECT_DOUBLE_EQ(history.value_at_or_before(300, 0.0), 0.7);
}

TEST(TimedCommandHistory, KeepsLatestEligibleSampleAfterRingWrap)
{
  TimedCommandHistory history;
  for (int64_t i = 1; i <= 10; ++i) {
    history.push(i * 100, static_cast<double>(i));
  }

  EXPECT_DOUBLE_EQ(history.value_at_or_before(850, -1.0), 8.0);
  EXPECT_DOUBLE_EQ(history.value_at_or_before(1000, -1.0), 10.0);
  EXPECT_DOUBLE_EQ(history.value_at_or_before(100, -1.0), -1.0);
}

TEST(TimedCommandHistory, HandlesOutOfOrderCallbackDeliveryByTimestamp)
{
  TimedCommandHistory history;
  history.push(300, 0.3);
  history.push(100, 0.1);
  history.push(200, 0.2);

  EXPECT_DOUBLE_EQ(history.value_at_or_before(250, 0.0), 0.2);
  EXPECT_DOUBLE_EQ(history.value_at_or_before(350, 0.0), 0.3);
}

}  // namespace
}  // namespace f1tenth_localization

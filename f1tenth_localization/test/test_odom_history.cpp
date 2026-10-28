#include "gpu_amcl_cpp/helpers/odom_history.hpp"

#include <deque>
#include <gtest/gtest.h>

namespace
{
struct Sample {long stamp;};
using gpu_amcl_cpp::odom_history::bracket;

TEST(OdomHistory, FutureScanWaitsForItsOwnOdometryEpoch)
{
  std::deque<Sample> samples{{100}, {125}};
  EXPECT_EQ(bracket(samples, 150L).first, nullptr);
  samples.push_back({150});
  const auto match = bracket(samples, 150L);
  ASSERT_NE(match.first, nullptr);
  EXPECT_EQ(match.first, match.second);
  EXPECT_EQ(match.first->stamp, 150);
}

TEST(OdomHistory, NeverSubstitutesAnEndpointOutsideHistory)
{
  const std::deque<Sample> samples{{100}, {125}, {150}};
  EXPECT_EQ(bracket(samples, 99L).first, nullptr);
  EXPECT_EQ(bracket(samples, 151L).first, nullptr);
}

TEST(OdomHistory, UsesAdjacentSamplesForInterpolation)
{
  const std::deque<Sample> samples{{100}, {125}, {150}, {175}};
  const auto match = bracket(samples, 140L);
  ASSERT_NE(match.first, nullptr);
  EXPECT_EQ(match.first->stamp, 125);
  EXPECT_EQ(match.second->stamp, 150);
  for (const auto & sample : samples) {
    const auto exact = bracket(samples, sample.stamp);
    EXPECT_EQ(exact.first, &sample);
    EXPECT_EQ(exact.second, &sample);
  }
}

TEST(OdomHistory, HandlesEmptyAndSingleSampleHistories)
{
  std::deque<Sample> samples;
  EXPECT_EQ(bracket(samples, 100L).first, nullptr);
  samples.push_back({100});
  EXPECT_EQ(bracket(samples, 99L).first, nullptr);
  EXPECT_EQ(bracket(samples, 101L).first, nullptr);
  EXPECT_EQ(bracket(samples, 100L).first, &samples.front());
}
}  // namespace

#include "gpu_amcl_cpp/helpers/scan_validity.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {
struct Scan {
  struct Header { long stamp{}; std::string frame_id{"laser"}; } header;
  float angle_min{-1.0f}, angle_max{1.0f}, angle_increment{0.1f};
  float range_min{0.1f}, range_max{10.0f};
  std::vector<float> ranges{1.0f, std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(), 2.0f};
};

TEST(ScanValidity, FreshStampDoesNotChangeHeldMeasurement)
{
  Scan first, second = first;
  second.header.stamp = 25;
  EXPECT_TRUE(gpu_amcl_cpp::scan_validity::same_observation(first, second));
  second.ranges.back() = std::nextafter(second.ranges.back(), 3.0f);
  EXPECT_FALSE(gpu_amcl_cpp::scan_validity::same_observation(first, second));
  second = first;
  second.angle_increment *= 2.0f;
  EXPECT_FALSE(gpu_amcl_cpp::scan_validity::same_observation(first, second));
  second = first;
  second.header.frame_id = "other_laser";
  EXPECT_FALSE(gpu_amcl_cpp::scan_validity::same_observation(first, second));
  first.ranges.clear(); second.ranges.clear();
  EXPECT_FALSE(gpu_amcl_cpp::scan_validity::same_observation(first, second));
}

TEST(ScanValidity, CountsOnlyActualUsableRangeEvidence)
{
  const Scan scan;
  const auto counts = gpu_amcl_cpp::scan_validity::count_sampled_valid_ranges(
      scan.ranges, 4, 0.1, 10.0);
  EXPECT_EQ(counts.sampled, 4u);
  EXPECT_EQ(counts.valid, 2u);
}
}  // namespace

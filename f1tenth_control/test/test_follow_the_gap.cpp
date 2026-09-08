#include <gtest/gtest.h>

#include "algorithms/follow_the_gap.hpp"

#include <cmath>
#include <vector>

namespace {

using f1tenth_control::FTGConfig;
using f1tenth_control::FollowTheGap;

constexpr std::size_t kSamples = 181;
constexpr double kPi = 3.14159265358979323846;
constexpr double kAngleMin = -kPi / 2.0;
constexpr double kAngleIncrement = kPi / 180.0;
constexpr double kAngleMax = kPi / 2.0;

std::vector<float> scanWithOpening(double lower, double upper,
                                   float wall = 0.10F) {
  std::vector<float> ranges(kSamples, wall);
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    const double angle = kAngleMin + static_cast<double>(index) * kAngleIncrement;
    if (angle >= lower && angle <= upper) {
      ranges[index] = 8.0F;
    }
  }
  return ranges;
}

FTGConfig testConfig() {
  FTGConfig config;
  config.max_speed = 1.0;
  config.min_speed = 0.2;
  config.max_steering = 0.5236;
  config.steering_gain = 3.2;
  config.max_steering_rate = 100.0;
  config.target_ema_alpha = 1.0;
  config.heading_weight = 0.2;
  config.score_power = 2.0;
  config.clearance_cone_scale = 1.0;
  config.min_score_range = 0.25;
  config.emergency_brake_distance = 0.10;
  config.footprint_clearance = 0.0;
  config.side_recovery_distance = 0.10;
  config.side_recovery_max_angle = 1.20;
  config.disparity_threshold = 100.0;
  config.wall_margin = 0.0;
  config.side_safety_margin = 0.0;
  config.lidar_config.range_min = 0.06;
  config.lidar_config.range_max = 10.0;
  config.lidar_config.angle_min = kAngleMin;
  config.lidar_config.angle_max = kAngleMax;
  config.lidar_config.apply_median_filter = false;
  return config;
}

}  // namespace

TEST(FollowTheGapRegression, AllowsFullConfiguredSteering) {
  auto config = testConfig();
  config.select_single_gap = true;
  FollowTheGap controller(config);

  const auto output = controller.compute(
      scanWithOpening(0.55, 1.45), kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_GT(output.command.steering_angle, 0.50);
  EXPECT_LE(std::abs(output.command.steering_angle), config.max_steering);
  EXPECT_LE(std::abs(output.raw_steering), config.max_steering);
}

TEST(FollowTheGapRegression, RecoversFromFrontSideBeamAcrossFullSector) {
  auto config = testConfig();
  config.side_recovery_distance = 0.60;
  config.side_recovery_max_angle = 1.20;
  FollowTheGap controller(config);
  auto ranges = std::vector<float>(kSamples, 8.0F);
  const std::size_t beam = static_cast<std::size_t>(
      (1.0 - kAngleMin) / kAngleIncrement);
  ranges[beam] = 0.50F;

  const auto output = controller.compute(
      ranges, kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_TRUE(output.side_recovery);
  EXPECT_LT(output.raw_steering, -config.side_recovery_min_steering);
  EXPECT_LE(std::abs(output.raw_steering), config.max_steering);
  EXPECT_LT(output.command.steering_angle, -config.side_recovery_min_steering);
}

TEST(FollowTheGapRegression, RecoveryNeverPublishesIntoObstacleDuringSlew) {
  auto config = testConfig();
  config.max_steering_rate = 0.10;
  config.side_recovery_distance = 0.60;
  config.side_recovery_max_angle = 1.20;
  config.ambiguous_front_recovery_sign = -1.0;
  FollowTheGap controller(config);

  auto clear = scanWithOpening(0.55, 1.45);
  const auto forward = controller.compute(
      clear, kAngleMin, kAngleMax, kAngleIncrement);
  EXPECT_GT(forward.command.steering_angle, 0.0);

  auto front_obstacle = std::vector<float>(kSamples, 8.0F);
  const std::size_t front_beam = static_cast<std::size_t>(
      (0.0 - kAngleMin) / kAngleIncrement);
  front_obstacle[front_beam] = 0.50F;
  const auto recovery = controller.compute(
      front_obstacle, kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_TRUE(recovery.side_recovery);
  EXPECT_LT(recovery.raw_steering, 0.0);
  EXPECT_LT(recovery.command.steering_angle, 0.0);
}

TEST(FollowTheGapRegression, AnticipatesTurnBeforeSteeringSlewCatchesUp) {
  auto config = testConfig();
  config.max_speed = 2.0;
  config.min_speed = 0.2;
  config.max_steering_rate = 0.10;
  config.steer_slowdown_gain = 1.0;
  FollowTheGap controller(config);

  const auto turn = scanWithOpening(0.45, 1.45);
  const auto output = controller.compute(
      turn, kAngleMin, kAngleMax, kAngleIncrement);

  // The first command is rate-limited near zero, but the scan already asks
  // for the configured corner steering.  Speed must use that future demand.
  EXPECT_LT(std::abs(output.command.steering_angle), 0.10);
  EXPECT_GT(std::abs(output.raw_steering), 0.50);
  EXPECT_LT(output.command.speed, 0.90);
}

TEST(FollowTheGapRegression, RecoveryRequiresConfirmedClearScans) {
  auto config = testConfig();
  config.side_recovery_distance = 0.60;
  config.recovery_clear_confirm_cycles = 3;
  config.lock_recovery_side_until_clear = true;
  FollowTheGap controller(config);

  auto obstacle = std::vector<float>(kSamples, 8.0F);
  const std::size_t beam = static_cast<std::size_t>(
      (1.0 - kAngleMin) / kAngleIncrement);
  obstacle[beam] = 0.50F;
  const auto initial = controller.compute(
      obstacle, kAngleMin, kAngleMax, kAngleIncrement);
  ASSERT_TRUE(initial.side_recovery);
  ASSERT_LT(initial.recovery_steering_sign, 0.0);

  const auto first_clear = controller.compute(
      std::vector<float>(kSamples, 8.0F), kAngleMin, kAngleMax, kAngleIncrement);
  const auto second_clear = controller.compute(
      std::vector<float>(kSamples, 8.0F), kAngleMin, kAngleMax, kAngleIncrement);
  EXPECT_TRUE(first_clear.side_recovery);
  EXPECT_TRUE(second_clear.side_recovery);
  EXPECT_LT(first_clear.command.steering_angle, 0.0);
  EXPECT_LT(second_clear.command.steering_angle, 0.0);
}

TEST(FollowTheGapRegression, VirtualFrontInflationRejectsInnerCornerClearance) {
  auto base_config = testConfig();
  base_config.side_recovery_distance = 0.60;
  auto ranges = std::vector<float>(kSamples, 8.0F);
  const std::size_t beam = static_cast<std::size_t>(
      (0.50 - kAngleMin) / kAngleIncrement);
  ranges[beam] = 0.21F;

  FollowTheGap base_controller(base_config);
  const auto base_output = base_controller.compute(
      ranges, kAngleMin, kAngleMax, kAngleIncrement);

  auto inflated_config = base_config;
  inflated_config.virtual_front_inflation = 0.08;
  FollowTheGap inflated_controller(inflated_config);
  const auto inflated_output = inflated_controller.compute(
      ranges, kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_FALSE(base_output.footprint_clearance_limited);
  EXPECT_TRUE(inflated_output.footprint_clearance_limited);
  EXPECT_TRUE(inflated_output.side_recovery);
  EXPECT_LE(inflated_output.command.speed, inflated_config.min_speed);
}

TEST(FollowTheGapRegression, RecoveryCanChangeSideAfterConfirmedCorner) {
  auto config = testConfig();
  config.side_recovery_distance = 0.60;
  config.recovery_switch_confirm_cycles = 3;
  FollowTheGap controller(config);

  auto positive_side = std::vector<float>(kSamples, 8.0F);
  auto negative_side = std::vector<float>(kSamples, 8.0F);
  const std::size_t positive_beam = static_cast<std::size_t>(
      (0.60 - kAngleMin) / kAngleIncrement);
  const std::size_t negative_beam = static_cast<std::size_t>(
      (-0.60 - kAngleMin) / kAngleIncrement);
  positive_side[positive_beam] = 0.45F;
  negative_side[negative_beam] = 0.45F;

  const auto first = controller.compute(
      positive_side, kAngleMin, kAngleMax, kAngleIncrement);
  const auto second = controller.compute(
      negative_side, kAngleMin, kAngleMax, kAngleIncrement);
  const auto third = controller.compute(
      negative_side, kAngleMin, kAngleMax, kAngleIncrement);
  const auto switched = controller.compute(
      negative_side, kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_LT(first.raw_steering, 0.0);
  EXPECT_LT(second.raw_steering, 0.0);
  EXPECT_LT(third.raw_steering, 0.0);
  EXPECT_GT(switched.raw_steering, 0.0);
}

TEST(FollowTheGapRegression, MappingRecoveryHoldsSideUntilClear) {
  auto config = testConfig();
  config.side_recovery_distance = 0.60;
  config.recovery_switch_confirm_cycles = 3;
  config.lock_recovery_side_until_clear = true;
  FollowTheGap controller(config);

  auto positive_side = std::vector<float>(kSamples, 8.0F);
  auto negative_side = std::vector<float>(kSamples, 8.0F);
  const std::size_t positive_beam = static_cast<std::size_t>(
      (0.60 - kAngleMin) / kAngleIncrement);
  const std::size_t negative_beam = static_cast<std::size_t>(
      (-0.60 - kAngleMin) / kAngleIncrement);
  positive_side[positive_beam] = 0.45F;
  negative_side[negative_beam] = 0.45F;

  const auto first = controller.compute(
      positive_side, kAngleMin, kAngleMax, kAngleIncrement);
  controller.compute(negative_side, kAngleMin, kAngleMax, kAngleIncrement);
  controller.compute(negative_side, kAngleMin, kAngleMax, kAngleIncrement);
  const auto held = controller.compute(
      negative_side, kAngleMin, kAngleMax, kAngleIncrement);
  const auto still_held = controller.compute(
      positive_side, kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_LT(first.raw_steering, 0.0);
  EXPECT_LT(held.raw_steering, 0.0);
  EXPECT_LT(still_held.raw_steering, 0.0);
}

TEST(FollowTheGapRegression, InflatedNearFrontCornerUsesSideClearance) {
  auto config = testConfig();
  config.side_recovery_distance = 0.60;
  config.virtual_front_inflation = 0.08;
  config.recovery_switch_confirm_cycles = 20;
  FollowTheGap controller(config);

  auto positive_side = std::vector<float>(kSamples, 8.0F);
  const std::size_t positive_beam = static_cast<std::size_t>(
      (0.60 - kAngleMin) / kAngleIncrement);
  const std::size_t negative_beam = static_cast<std::size_t>(
      (-0.60 - kAngleMin) / kAngleIncrement);
  const std::size_t front_beam = static_cast<std::size_t>(
      (-0.10 - kAngleMin) / kAngleIncrement);
  positive_side[positive_beam] = 0.45F;
  const auto first = controller.compute(
      positive_side, kAngleMin, kAngleMax, kAngleIncrement);

  auto near_front = std::vector<float>(kSamples, 8.0F);
  near_front[negative_beam] = 0.35F;
  near_front[front_beam] = 0.20F;
  const auto corrected = controller.compute(
      near_front, kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_LT(first.raw_steering, 0.0);
  EXPECT_TRUE(corrected.footprint_clearance_limited);
  EXPECT_TRUE(corrected.side_recovery);
  EXPECT_GT(corrected.raw_steering, 0.0);
}

TEST(FollowTheGapRegression, DoesNotEraseTurnDirectionThroughStraightScan) {
  auto config = testConfig();
  config.select_single_gap = true;
  config.gap_switch_confirm_cycles = 4;
  FollowTheGap controller(config);

  const auto left = scanWithOpening(0.35, 1.45);
  const auto straight = scanWithOpening(-0.12, 0.12);
  const auto right = scanWithOpening(-1.45, -0.35);

  const auto first = controller.compute(
      left, kAngleMin, kAngleMax, kAngleIncrement);
  controller.compute(straight, kAngleMin, kAngleMax, kAngleIncrement);
  const auto first_opposite = controller.compute(
      right, kAngleMin, kAngleMax, kAngleIncrement);
  const auto second_opposite = controller.compute(
      right, kAngleMin, kAngleMax, kAngleIncrement);
  const auto third_opposite = controller.compute(
      right, kAngleMin, kAngleMax, kAngleIncrement);
  const auto accepted_opposite = controller.compute(
      right, kAngleMin, kAngleMax, kAngleIncrement);
  auto settled_opposite = accepted_opposite;
  for (int index = 0; index < 8; ++index) {
    settled_opposite = controller.compute(
        right, kAngleMin, kAngleMax, kAngleIncrement);
  }

  EXPECT_GT(first.command.steering_angle, 0.0);
  EXPECT_GT(first_opposite.command.steering_angle, 0.0);
  EXPECT_GT(second_opposite.command.steering_angle, 0.0);
  EXPECT_GT(third_opposite.command.steering_angle, 0.0);
  EXPECT_LT(accepted_opposite.target_angle, 0.0);
  EXPECT_LT(settled_opposite.command.steering_angle, 0.0);
}

TEST(FollowTheGapRegression, CenteredScansDoNotErasePendingGapSwitch) {
  auto config = testConfig();
  config.select_single_gap = true;
  config.gap_switch_confirm_cycles = 4;
  FollowTheGap controller(config);

  const auto left = scanWithOpening(0.35, 1.45);
  const auto straight = scanWithOpening(-0.08, 0.08);
  const auto right = scanWithOpening(-1.45, -0.35);

  const auto first = controller.compute(
      left, kAngleMin, kAngleMax, kAngleIncrement);
  controller.compute(straight, kAngleMin, kAngleMax, kAngleIncrement);
  controller.compute(straight, kAngleMin, kAngleMax, kAngleIncrement);
  controller.compute(right, kAngleMin, kAngleMax, kAngleIncrement);
  const auto still_pending = controller.compute(
      right, kAngleMin, kAngleMax, kAngleIncrement);

  EXPECT_GT(first.target_angle, 0.12);
  EXPECT_GT(still_pending.target_angle, 0.0);
}

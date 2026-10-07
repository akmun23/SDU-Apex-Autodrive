#include <cmath>

#include <gtest/gtest.h>

#include "f1tenth_localization/odometry_observer.hpp"

namespace f1tenth_localization
{
namespace
{

TEST(OdometryObserverDiagnostics, SeparatesPreWheelPredictionFromFusedSpeed)
{
  OdometryObserver observer;
  OdometryObservation sample;
  sample.stamp_s = 1.0;
  auto estimate = observer.update(sample);
  EXPECT_FALSE(estimate.pre_wheel_update_speed_valid);

  sample.stamp_s += 0.025;
  sample.left_angle_rad = 0.30;
  sample.right_angle_rad = 0.30;
  sample.ax_mps2 = 1.0;
  estimate = observer.update(sample);

  ASSERT_TRUE(estimate.pre_wheel_update_speed_valid);
  EXPECT_NEAR(estimate.pre_wheel_update_speed_mps, 0.025, 1.0e-9);
  EXPECT_GT(estimate.speed_mps, estimate.pre_wheel_update_speed_mps + 0.5);
}

TEST(OdometryObserverDiagnostics, InvalidatesReferenceOnTurnEntry)
{
  OdometryObserver observer;
  OdometryObservation sample;
  sample.stamp_s = 1.0;
  observer.update(sample);

  sample.stamp_s += 0.025;
  sample.left_angle_rad = 0.30;
  sample.right_angle_rad = 0.30;
  sample.yaw_rate_radps = 1.0;
  const auto entry = observer.update(sample);
  EXPECT_TRUE(entry.turn_mode);
  EXPECT_FALSE(entry.pre_wheel_update_speed_valid);

  sample.stamp_s += 0.025;
  sample.left_angle_rad += 0.30;
  sample.right_angle_rad += 0.30;
  const auto established_turn = observer.update(sample);
  EXPECT_TRUE(established_turn.turn_mode);
  EXPECT_TRUE(established_turn.pre_wheel_update_speed_valid);
}

}  // namespace
}  // namespace f1tenth_localization

#include "f1tenth_localization/physx_odom_actuation.hpp"

#include <gtest/gtest.h>

namespace f1tenth_localization::detail
{
namespace
{

TEST(PhysxOdomActuation, UsesQueuedBrakeDespitePreviousDriveFeedback)
{
  float steering_internal_deg = 0.0f;
  const auto actuation = make_physx_packet_actuation(
    steering_internal_deg, 0.0f, 0.0f, true, 0.0);

  EXPECT_FLOAT_EQ(actuation.throttle_normalized, 0.0f);
}

TEST(PhysxOdomActuation, UsesQueuedDriveDespitePreviousBrakeFeedback)
{
  float steering_internal_deg = 0.0f;
  const auto actuation = make_physx_packet_actuation(
    steering_internal_deg, 0.0f, 0.56f, true, 0.0);

  EXPECT_FLOAT_EQ(actuation.throttle_normalized, 0.56f);
}

TEST(PhysxOdomActuation, FeedbackAnchorsSteeringThenPendingCommandAdvancesIt)
{
  float steering_internal_deg = 20.0f;
  const auto actuation = make_physx_packet_actuation(
    steering_internal_deg, 1.0f, 0.25f, true, 0.1);

  EXPECT_NEAR(actuation.steering_angles_deg.front(), -5.912f, 0.002f);
  EXPECT_NEAR(actuation.steering_angles_deg.back(), -10.312f, 0.01f);
  EXPECT_NEAR(steering_internal_deg, actuation.steering_angles_deg.back(), 1.0e-6f);
}

TEST(PhysxOdomActuation, WithoutFeedbackSteeringContinuesFromModelState)
{
  float steering_internal_deg = 0.0f;
  const auto actuation = make_physx_packet_actuation(
    steering_internal_deg, -1.0f, 0.4f, false, 0.0);

  EXPECT_NEAR(actuation.steering_angles_deg.back(), 4.58365f, 0.01f);
  EXPECT_FLOAT_EQ(actuation.throttle_normalized, 0.4f);
}

}  // namespace
}  // namespace f1tenth_localization::detail

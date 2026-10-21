#include "mpc_drive_command.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace f1tenth_mpc {
namespace {

constexpr double kSteeringLimit = 0.52359877559829887308;

TEST(MpcDriveCommandEnvelope, ClampsOnlyPublicationRoundoff)
{
    MpcDriveCommand bounded{};
    ASSERT_TRUE(bound_mpc_drive_command(
        kSteeringLimit + 1.44e-8, 1.0 + 5.0e-7,
        kSteeringLimit, &bounded));
    EXPECT_DOUBLE_EQ(bounded.steering_target_rad, kSteeringLimit);
    EXPECT_DOUBLE_EQ(bounded.throttle_normalized, 1.0);
}

TEST(MpcDriveCommandEnvelope, RejectsPhysicalEnvelopeViolation)
{
    MpcDriveCommand bounded{};
    EXPECT_FALSE(bound_mpc_drive_command(
        kSteeringLimit + 2.0e-6, 0.5, kSteeringLimit, &bounded));
    EXPECT_FALSE(bound_mpc_drive_command(
        0.0, static_cast<double>(MPC_THROTTLE_MIN_DRIVE) - 2.0e-6,
        kSteeringLimit, &bounded));
    EXPECT_FALSE(bound_mpc_drive_command(
        0.0, 0.0, kSteeringLimit, &bounded));
    EXPECT_FALSE(bound_mpc_drive_command(
        0.0, 1.0 + 2.0e-6, kSteeringLimit, &bounded));
}

TEST(MpcDriveCommandEnvelope, RejectsNonFiniteInputsAndInvalidLimit)
{
    MpcDriveCommand bounded{};
    EXPECT_FALSE(bound_mpc_drive_command(
        std::numeric_limits<double>::quiet_NaN(), 0.5,
        kSteeringLimit, &bounded));
    EXPECT_FALSE(bound_mpc_drive_command(
        0.0, std::numeric_limits<double>::infinity(),
        kSteeringLimit, &bounded));
    EXPECT_FALSE(bound_mpc_drive_command(0.0, 0.5, 0.0, &bounded));
    EXPECT_FALSE(bound_mpc_drive_command(0.0, 0.5, kSteeringLimit, nullptr));
}

}  // namespace
}  // namespace f1tenth_mpc

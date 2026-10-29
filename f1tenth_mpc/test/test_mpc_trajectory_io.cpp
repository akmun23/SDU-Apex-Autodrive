#include "mpc_trajectory_io.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace {
std::string triangle(const std::string &suffix = "")
{
    return "0,0,0,0,0,5,0,1,1" + suffix + "\n" +
           "1,1,0,1.57,0,6,0,1,1" + suffix + "\n" +
           "2,1,1,3.14,0,7,0,1,1" + suffix + "\n";
}
}

TEST(MpcTrajectoryIo, OrdinaryReferenceUnchanged)
{
    std::istringstream input("# s_m,x_m,...\n" + triangle());
    double length = 0.;
    const auto points = f1tenth_mpc::read_mpc_trajectory(input, &length);
    ASSERT_EQ(points.size(), 3u);
    EXPECT_FALSE(points[0].source_witness_valid);
    EXPECT_DOUBLE_EQ(points[1].speed, 6.);
    EXPECT_NEAR(length, 2.+std::sqrt(2.), 1.e-12);
}

TEST(MpcTrajectoryIo, WitnessKeepsEveryFieldForRuntimeAndReplay)
{
    std::istringstream input("# sdu_apex_source_witness/v1\r\n" +
        triangle(",.03,.04,-2.1,-.2,-.25,.15"));
    double length = 0.;
    const auto points = f1tenth_mpc::read_mpc_trajectory(input, &length);
    for (const auto &point : points) {
        EXPECT_TRUE(point.source_witness_valid);
        EXPECT_DOUBLE_EQ(point.source_heading_error, .03);
        EXPECT_DOUBLE_EQ(point.source_v_rear, .04);
        EXPECT_DOUBLE_EQ(point.source_yaw_rate, -2.1);
        EXPECT_DOUBLE_EQ(point.source_physical_steering, -.2);
        EXPECT_DOUBLE_EQ(point.source_steering_target, -.25);
        EXPECT_DOUBLE_EQ(point.source_throttle, .15);
    }
}

TEST(MpcTrajectoryIo, InvalidRowsCannotBeSilentlyDroppedOrTruncated)
{
    const auto source = "# sdu_apex_source_witness/v1\n";
    for (const auto &text : {
            triangle(",0,0,0,0,0,.2"),  // unmarked source fields
            source+triangle(),
            triangle()+"3,0\n",
            triangle()+"3,0,1,0,0,5,0,1,1,\n",
            triangle()+"3,0,1,0,0,5junk,0,1,1\n",
            triangle()+"3,0,1,0,0,nan,0,1,1\n",
            triangle()+source,
            source+triangle(",0,0,0,0,0,0")}) {
        std::istringstream input(text);
        double length = 0.;
        EXPECT_THROW(f1tenth_mpc::read_mpc_trajectory(input, &length),
                     std::runtime_error) << text;
    }
}

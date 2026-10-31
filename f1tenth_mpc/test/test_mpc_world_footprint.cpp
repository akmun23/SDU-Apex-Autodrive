#include "mpc_world_footprint.hpp"
#include <gtest/gtest.h>
#include <vector>
using f1tenth_mpc::MpcBodyPose;
using f1tenth_mpc::MpcWorldFootprint;
TEST(MpcWorldFootprint, RejectsUnknownCellsAndOutsideMap) {
    std::vector<int8_t> data(100*100,0);
    data[50*100+50]=-1;
    const auto grid=MpcWorldFootprint::from_occupancy(100,100,.05,0,0,0,data);
    ASSERT_TRUE(grid);
    EXPECT_LT(grid->point_clearance_lower_bound(2.525,2.525),0);
    EXPECT_LT(grid->point_clearance_lower_bound(-.01,2),0);
    EXPECT_GT(grid->point_clearance_lower_bound(1.,1.),.5);
}
TEST(MpcWorldFootprint, BodyAndSweepDetectRealObstacle) {
    std::vector<int8_t> data(100*100,0);
    for(int j=0;j<100;++j)data[j*100+55]=100;
    const auto grid=MpcWorldFootprint::from_occupancy(100,100,.05,0,0,0,data);
    ASSERT_TRUE(grid);
    double slack=0.;
    EXPECT_TRUE(grid->interpolated_sweep_passes(
        MpcBodyPose{1.,2.5,0},MpcBodyPose{1.2,2.5,0},.02,&slack));
    EXPECT_GT(slack,0);
    EXPECT_FALSE(grid->interpolated_sweep_passes(
        MpcBodyPose{2.0,2.5,0},MpcBodyPose{2.5,2.5,0},.02,&slack));
}
TEST(MpcWorldFootprint, MapYawAndResolutionTransform) {
    std::vector<int8_t> data(60*60,0);
    data[20*60+20]=100;
    const auto grid=MpcWorldFootprint::from_occupancy(
        60,60,.025,-1.,-1.,1.5707963267948966,data);
    ASSERT_TRUE(grid);
    const double x=-1.-(20.5*.025),y=-1.+20.5*.025;
    EXPECT_LT(grid->point_clearance_lower_bound(x,y),0);
    EXPECT_GT(grid->point_clearance_lower_bound(x-.2,y+.2),.1);
}

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

TEST(MpcWorldFootprint, LookupIsConservativeAgainstExactOccupiedSquares)
{
    constexpr int w=72,h=72;
    constexpr double res=.025;
    std::vector<int8_t> cells(w*h,0);
    for (int row=14;row<60;++row) {
        cells[row*w+16]=100;
        cells[row*w+61]=100;
    }
    for (int row=28;row<35;++row)
        for (int col=29;col<34;++col)
            cells[row*w+col]=100;
    const auto map=MpcWorldFootprint::from_occupancy(
        w,h,res,0.,0.,0.,cells);
    ASSERT_TRUE(map);
    // Deterministic fractional coordinates exercise all 64x64 table bins.
    // Exact distance-to-occupied-SQUARE is an upper bound on any correctly
    // conservative precomputed Euclidean field certificate.
    for (int n=0;n<1000;++n) {
        const double x=.06+1.65*double((n*197)%997)/996.;
        const double y=.06+1.65*double((n*431)%991)/990.;
        double exact=INFINITY;
        for (int row=0;row<h;++row) {
            for (int col=0;col<w;++col) {
                if (row!=0 && col!=0 && row!=h-1 && col!=w-1 &&
                    cells[row*w+col]==0) continue;
                const double dx=std::max({col*res-x,0.,x-(col+1)*res});
                const double dy=std::max({row*res-y,0.,y-(row+1)*res});
                exact=std::min(exact,std::hypot(dx,dy));
            }
        }
        const double bound=map->point_clearance_lower_bound(x,y);
        ASSERT_TRUE(std::isfinite(bound));
        EXPECT_LE(bound,exact+1.e-6) << "sample " << n;
        EXPECT_GT(bound,exact-.1) << "sample " << n;
    }
}

TEST(MpcWorldFootprint, BroadphaseCannotMissAThinMapWall)
{
    constexpr int w=140,h=140;
    std::vector<int8_t> cells(w*h,0);
    for (int y=1;y<h-1;++y) cells[y*w+75]=100;
    const auto map=MpcWorldFootprint::from_occupancy(
        w,h,.025,0.,0.,0.,cells);
    ASSERT_TRUE(map);
    double slack=0.;
    EXPECT_TRUE(map->interpolated_sweep_passes(
        {1.,2.,0.},{1.04,2.,0.},.05,&slack));
    EXPECT_GE(slack,0.);
    EXPECT_FALSE(map->interpolated_sweep_passes(
        {1.40,2.,0.},{1.55,2.,0.},.05,&slack));
    EXPECT_LT(slack,0.);
}

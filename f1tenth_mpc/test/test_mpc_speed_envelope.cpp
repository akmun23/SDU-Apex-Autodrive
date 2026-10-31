#include "mpc_speed_envelope.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

TEST(MpcSpeedEnvelope, LimitsCurvatureAndAnticipatesBraking)
{
    std::vector<MpcTrajectorySample_t> p(5);
    for(int i=0;i<5;++i){p[i].s=i;p[i].speed=10.;p[i].source_witness_valid=1;}
    p[3].curvature=1.;
    const auto summary=f1tenth_mpc::apply_curvature_speed_envelope(p,5.,4.,2.);
    EXPECT_EQ(summary.reduced_points,5u);
    EXPECT_TRUE(summary.witness_invalidated);
    EXPECT_NEAR(p[3].speed,2.,1e-9);
    EXPECT_NEAR(p[2].speed,std::sqrt(8.),1e-9);
    EXPECT_NEAR(p[1].speed,std::sqrt(12.),1e-9);
    EXPECT_NEAR(p[0].speed,4.,1e-9);
    EXPECT_NEAR(p[4].speed,std::sqrt(20.),1e-9);
    for(const auto &v:p) EXPECT_EQ(v.source_witness_valid,0);
}

TEST(MpcSpeedEnvelope, ClosedLoopSeamIsBounded)
{
    std::vector<MpcTrajectorySample_t> p(4);
    for(int i=0;i<4;++i){p[i].s=i;p[i].speed=15.;}
    p[0].curvature=1.;
    f1tenth_mpc::apply_curvature_speed_envelope(p,4.,4.,3.);
    EXPECT_NEAR(p[3].speed,std::sqrt(10.),1e-9);
    EXPECT_NEAR(p[2].speed,4.,1e-9);
}

TEST(MpcSpeedEnvelope, NoModificationRetainsWitness)
{
    std::vector<MpcTrajectorySample_t> p(3);
    for(int i=0;i<3;++i){p[i].s=i;p[i].speed=2.;p[i].source_witness_valid=1;}
    const auto r=f1tenth_mpc::apply_curvature_speed_envelope(p,3.,6.,4.);
    EXPECT_EQ(r.reduced_points,0u);
    EXPECT_FALSE(r.witness_invalidated);
    for(const auto &v:p) EXPECT_EQ(v.source_witness_valid,1);
}

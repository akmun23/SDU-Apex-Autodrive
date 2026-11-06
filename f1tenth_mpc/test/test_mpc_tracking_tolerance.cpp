#include "mpc_tracking_tolerance.h"
#include <gtest/gtest.h>
#include <cmath>
TEST(MpcTrackingTolerance, ZeroPreservesOriginalQuadratic) {
    float target=0;
    for (float error : {-0.04f, 0.0f, 0.02f, 0.30f}) {
        EXPECT_EQ(mpc_tracking_local_target(error,0.1f,0.f,&target),1);
        EXPECT_FLOAT_EQ(target,0.1f);
        EXPECT_FLOAT_EQ(mpc_tracking_excess(error,0.f),error);
    }
}
TEST(MpcTrackingTolerance, DeviationInsideDeadzoneHasNoTrackingPenalty) {
    float target=999;
    for (float error : {-0.019f, 0.f, 0.019f}) {
        EXPECT_EQ(mpc_tracking_local_target(0.1f+error,0.1f,0.02f,&target),0);
        EXPECT_NEAR(mpc_tracking_excess(error,0.02f),0.f,1e-6f);
    }
}
TEST(MpcTrackingTolerance, ActiveErrorMatchesExactMeritDerivative) {
    constexpr float ref=0.13f, tolerance=0.03f, weight=6000.f;
    for (float error : {-0.10f,-0.05f,0.05f,0.10f}) {
        float target=0;
        const float nominal=ref+error;
        ASSERT_EQ(mpc_tracking_local_target(nominal,ref,tolerance,&target),1);
        const float gradient_local=2*weight*(nominal-target);
        const float gradient_exact=2*weight*mpc_tracking_excess(error,tolerance);
        EXPECT_NEAR(gradient_local,gradient_exact,0.0005f);
        EXPECT_NEAR(weight*(nominal-target)*(nominal-target),
            weight*std::pow(mpc_tracking_excess(error,tolerance),2),1e-5f);
    }
}
TEST(MpcTrackingTolerance, HeadingAndLateralAreIndependent) {
    float target=0;
    EXPECT_EQ(mpc_tracking_local_target(.02f,0.f,.025f,&target),0);
    EXPECT_EQ(mpc_tracking_local_target(.06f,0.f,.015f,&target),1);
    EXPECT_NEAR(target,.015f,1e-6f);
    EXPECT_EQ(mpc_tracking_local_target(-.06f,0.f,.015f,&target),1);
    EXPECT_NEAR(target,-.015f,1e-6f);
}

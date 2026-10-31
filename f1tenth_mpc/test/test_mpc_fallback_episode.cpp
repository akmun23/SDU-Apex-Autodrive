#include "mpc_fallback_episode.hpp"
#include <gtest/gtest.h>
using f1tenth_mpc::FallbackEpisode;
TEST(MpcFallbackEpisode, IsolatedFailureCanResume) {
    FallbackEpisode a;
    EXPECT_FALSE(a.must_reseed_nominal());
    a.record_short_action();
    EXPECT_TRUE(a.must_reseed_nominal());
    EXPECT_FALSE(a.requires_full_horizon(2));
    a.record_full_action();
    EXPECT_EQ(a.consecutive_short_actions(),0);
    EXPECT_EQ(a.recent_short_actions(),1);
    EXPECT_FALSE(a.requires_full_horizon(2));
}
TEST(MpcFallbackEpisode, AlternatingSourcesAndIntermittentConvergenceDoNotResetExposure) {
    FallbackEpisode a;
    a.record_short_action(); // retained
    a.record_short_action(); // candidate
    EXPECT_TRUE(a.requires_full_horizon(2));
    a.record_full_action();
    EXPECT_EQ(a.consecutive_short_actions(),0);
    a.record_short_action();
    EXPECT_TRUE(a.requires_full_horizon(2)); // recent count is three
    a.reset();
    EXPECT_FALSE(a.must_reseed_nominal());
}
TEST(MpcFallbackEpisode, HistoryExpiresAfterSixteenFullActions) {
    FallbackEpisode a;
    a.record_short_action();
    a.record_short_action();
    a.record_short_action();
    EXPECT_TRUE(a.requires_full_horizon(2));
    for(int i=0;i<16;++i)a.record_full_action();
    EXPECT_EQ(a.recent_short_actions(),0);
    EXPECT_FALSE(a.requires_full_horizon(2));
}
TEST(MpcFallbackEpisode, ZeroLeavesOriginalBehaviorUnaffected) {
    FallbackEpisode a;
    for(int i=0;i<30;++i)a.record_short_action();
    EXPECT_FALSE(a.requires_full_horizon(0));
}

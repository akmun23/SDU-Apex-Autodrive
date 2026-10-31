#include "mpc_fallback_seed.hpp"
#include <gtest/gtest.h>
namespace {
TEST(MpcFallbackSeed, KeepsExactThreeStepPrefixOnlyAfterShortAction) {
    MpcRtiMemory_t memory{};
    mpc_rti_memory_reset(&memory);
    memory.nominal.valid=1;
    MpcModelControl_t original[30]{};
    for(int i=0;i<30;++i){
        original[i].steering_target_rad=0.001f*i;
        original[i].throttle_command=0.5f;
    }
    ASSERT_TRUE(mpc_rti_memory_set_warm_start_controls(&memory,original,30));
    f1tenth_mpc::FallbackEpisode episode;
    episode.record_short_action();
    f1tenth_mpc::reseed_after_bounded_fallback(&memory,episode,3);
    EXPECT_FALSE(memory.nominal.valid);
    EXPECT_TRUE(memory.warm_start_controls_valid);
    EXPECT_EQ(memory.warm_start_control_count,3);
    EXPECT_FLOAT_EQ(memory.warm_start_controls[1].steering_target_rad,.001f);
}
TEST(MpcFallbackSeed, DoesNotForgetLastFullPlanDespiteRecentFallback) {
    MpcRtiMemory_t memory{};
    mpc_rti_memory_reset(&memory);
    memory.nominal.valid=1;
    MpcModelControl_t original[30]{};
    ASSERT_TRUE(mpc_rti_memory_set_warm_start_controls(&memory,original,30));
    f1tenth_mpc::FallbackEpisode episode;
    episode.record_short_action();
    episode.record_short_action();
    episode.record_full_action();
    ASSERT_GE(episode.recent_short_actions(),2);
    f1tenth_mpc::reseed_after_bounded_fallback(&memory,episode,3);
    EXPECT_TRUE(memory.nominal.valid);
    EXPECT_EQ(memory.warm_start_control_count,30);
}
TEST(MpcFallbackSeed, MissingCertifiedPrefixNeverInventsIt) {
    MpcRtiMemory_t memory{};
    mpc_rti_memory_reset(&memory);
    memory.nominal.valid=1;
    f1tenth_mpc::FallbackEpisode episode;
    episode.record_short_action();
    f1tenth_mpc::reseed_after_bounded_fallback(&memory,episode,3);
    EXPECT_FALSE(memory.nominal.valid);
    EXPECT_FALSE(memory.warm_start_controls_valid);
    EXPECT_EQ(memory.warm_start_control_count,0);
}
}

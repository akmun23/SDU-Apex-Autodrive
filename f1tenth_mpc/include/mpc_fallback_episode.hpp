#pragma once

#include <algorithm>
#include <cstdint>

namespace f1tenth_mpc {

// Short-prefix certification is only 25-75ms; repeated reuse across adjacent
// 25ms commands can silently exceed its accuracy horizon. A single feasible
// full-horizon action resets the *consecutive* chain, but not the accumulated
// number of recent short prefixes: one intermittent good solve must not erase
// a long loss-of-convergence episode.
class FallbackEpisode {
public:
    void reset() {
        consecutive_short_actions_ = 0;
        recent_short_actions_ = 0;
        history_ = 0;
    }
    void record_short_action() {
        slide(true);
        consecutive_short_actions_ = std::min(consecutive_short_actions_+1,1000);
    }
    void record_full_action() {
        slide(false);
        consecutive_short_actions_ = 0;
    }
    int consecutive_short_actions() const {return consecutive_short_actions_;}
    int recent_short_actions() const {return recent_short_actions_;}
    bool must_reseed_nominal() const {
        return consecutive_short_actions_ > 0 || recent_short_actions_ >= 2;
    }
    bool requires_full_horizon(int maximum_short_actions) const {
        return maximum_short_actions > 0 &&
            (consecutive_short_actions_ >= maximum_short_actions ||
             recent_short_actions_ > maximum_short_actions);
    }
private:
    // 16 published actions at nominal 40Hz = 400ms. Constant-time shift and
    // popcount update; no allocation or scan in the control callback.
    void slide(bool short_prefix) {
        constexpr std::uint32_t dropped_mask=1u << 15;
        if (history_ & dropped_mask) --recent_short_actions_;
        history_ = ((history_ << 1) | (short_prefix ? 1u : 0u)) & 0xFFFFu;
        if (short_prefix) ++recent_short_actions_;
    }
    int consecutive_short_actions_{};
    int recent_short_actions_{};
    std::uint32_t history_{};
};
} // namespace f1tenth_mpc

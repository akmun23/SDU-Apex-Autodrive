#pragma once
// Preserve ONLY issued and exact-validated 75ms controls from a short-prefix
// recovery; discard unvalidated full-horizon tail. On a new fully accepted
// plan, do not erase its freshly validated nominal merely because an older
// fallback remains in the 16-action health window. Explicitly opt-in.
#include "mpc_rti.h"
#include "mpc_fallback_episode.hpp"
namespace f1tenth_mpc {
inline void reseed_after_bounded_fallback(MpcRtiMemory_t *memory,
    const FallbackEpisode &episode,int exact_prefix_horizon)
{
    if(!memory || !episode.must_reseed_nominal()) return;
    if(episode.consecutive_short_actions()==0) {
        // Last issued action came from a new exact valid 750ms horizon.
        // It supersedes the short plan's rejected tail.
        return;
    }
    memory->nominal.valid=0;
    if(memory->warm_start_controls_valid &&
       memory->warm_start_control_count>=exact_prefix_horizon &&
       exact_prefix_horizon>1) {
        // slot 0: issued command; slot 1..: verified actuator-delay suffix.
        // Solver shifts out slot 0, then synthesizes the unverified tail
        // directly from the reference instead of inheriting rejected controls.
        memory->warm_start_control_count=exact_prefix_horizon;
    } else {
        memory->warm_start_controls_valid=0;
        memory->warm_start_control_count=0;
    }
    mpc_rti_memory_reset_solver_state(memory);
}
} // namespace f1tenth_mpc

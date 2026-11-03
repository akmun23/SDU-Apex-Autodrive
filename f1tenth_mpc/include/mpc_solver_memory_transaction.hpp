#pragma once

#include "mpc_rti.h"
#include <cstdint>

namespace f1tenth_mpc {

/* Transaction around a 40Hz solve. A computed-but-unissued solution does
 * NOT own the previously published command, source nominal, or ADMM duals.
 * Taking a 22KB POD snapshot before the solve permits exact rollback after a
 * deadline miss, instead of discarding the last issued plan's warm state.
 * Always revalidate the certified action from the CURRENT source plant
 * before publication; this class only restores an optimization seed. */
class SolverMemoryTransaction {
public:
    void begin(const MpcRtiMemory_t &issued_memory,
               std::int64_t command_epoch_ns,bool issued_full_horizon_valid) {
        valid_ = issued_full_horizon_valid && command_epoch_ns>0 &&
            issued_memory.nominal.valid &&
            issued_memory.warm_start_controls_valid;
        if (valid_) {
            saved_ = issued_memory;
            epoch_ns_ = command_epoch_ns;
        } else epoch_ns_ = 0;
    }

    bool rollback(MpcRtiMemory_t *memory,std::int64_t command_epoch_ns) {
        if (!valid_ || !memory || command_epoch_ns!=epoch_ns_)return false;
        *memory = saved_;
        valid_=false;epoch_ns_=0;
        return true;
    }

    void clear() { valid_=false; epoch_ns_=0; }
private:
    MpcRtiMemory_t saved_{};
    std::int64_t epoch_ns_{};
    bool valid_{};
};
} // namespace f1tenth_mpc

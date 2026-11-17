#pragma once

#include "mpc_cycle.h"
#include <cstdint>

namespace f1tenth_mpc {

/* Snapshot a 40 Hz solve so a deadline miss restores the last issued plan's
 * warm state. Fresh-state solves validate each new plan independently. State
 * gaps may continue the validated issued plan at its original command epochs;
 * this transaction does not certify a computed but unissued plan. */
class SolverMemoryTransaction {
public:
    void begin(const MpcCycleMemory_t &issued_memory,
               std::int64_t command_epoch_ns,bool issued_full_horizon_valid) {
        valid_ = issued_full_horizon_valid && command_epoch_ns>0 &&
            issued_memory.nominal.valid &&
            issued_memory.warm_start_controls_valid;
        if (valid_) {
            saved_ = issued_memory;
            epoch_ns_ = command_epoch_ns;
        } else epoch_ns_ = 0;
    }

    bool rollback(MpcCycleMemory_t *memory,std::int64_t command_epoch_ns) {
        if (!valid_ || !memory || command_epoch_ns!=epoch_ns_)return false;
        *memory = saved_;
        valid_=false;epoch_ns_=0;
        return true;
    }

    void clear() { valid_=false; epoch_ns_=0; }
private:
    MpcCycleMemory_t saved_{};
    std::int64_t epoch_ns_{};
    bool valid_{};
};
} // namespace f1tenth_mpc

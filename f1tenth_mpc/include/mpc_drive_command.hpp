#ifndef MPC_DRIVE_COMMAND_HPP
#define MPC_DRIVE_COMMAND_HPP

#include "mpc_types.h"

#include <algorithm>
#include <cmath>

namespace f1tenth_mpc {

struct MpcDriveCommand {
    double steering_target_rad{};
    double throttle_normalized{};
};

/* Accept only the same 1e-6 numeric-roundoff tolerance used by the live
 * publisher, then clamp that representational overrun before publication. */
inline bool bound_mpc_drive_command(
    double steering_target_rad, double throttle_normalized,
    double steering_limit_rad, MpcDriveCommand * bounded)
{
    constexpr double kRoundoffTolerance = 1.0e-6;
    if (bounded == nullptr || !std::isfinite(steering_target_rad) ||
        !std::isfinite(throttle_normalized) ||
        !std::isfinite(steering_limit_rad) || steering_limit_rad <= 0.0 ||
        std::abs(steering_target_rad) >
            steering_limit_rad + kRoundoffTolerance ||
        throttle_normalized <
            static_cast<double>(MPC_THROTTLE_MIN_DRIVE) - kRoundoffTolerance ||
        throttle_normalized > 1.0 + kRoundoffTolerance) {
        return false;
    }
    bounded->steering_target_rad = std::clamp(
        steering_target_rad, -steering_limit_rad, steering_limit_rad);
    bounded->throttle_normalized = std::clamp(
        throttle_normalized,
        static_cast<double>(MPC_THROTTLE_MIN_DRIVE), 1.0);
    return true;
}

}  // namespace f1tenth_mpc

#endif

#ifndef MPC_AUTHORITY_POLICY_HPP
#define MPC_AUTHORITY_POLICY_HPP

#include <algorithm>
#include <cmath>

namespace f1tenth_mpc {

inline bool mpc_localization_covariance_good(
    double covariance_x,
    double covariance_y,
    double covariance_yaw,
    double maximum_xy,
    double maximum_yaw)
{
    return std::isfinite(covariance_x) && std::isfinite(covariance_y) &&
        std::isfinite(covariance_yaw) && covariance_x >= 0.0 &&
        covariance_y >= 0.0 && covariance_yaw >= 0.0 &&
        std::max(covariance_x, covariance_y) <= maximum_xy &&
        covariance_yaw <= maximum_yaw;
}

inline double mpc_select_current_steering(
    double modeled_command_rad,
    bool feedback_fresh,
    double feedback_rad,
    double maximum_steering_rad)
{
    if (!(maximum_steering_rad > 0.0) ||
        !std::isfinite(maximum_steering_rad)) {
        return 0.0;
    }
    double steering = modeled_command_rad;
    if (feedback_fresh && std::isfinite(feedback_rad))
        steering = feedback_rad;
    if (!std::isfinite(steering))
        steering = 0.0;
    return std::clamp(
        steering, -maximum_steering_rad, maximum_steering_rad);
}

inline double mpc_fallback_target_speed(
    bool has_accepted_command,
    double previous_command_mps,
    double observed_speed_mps,
    double requested_speed_mps,
    double local_raceline_cap_mps,
    double active_speed_ceiling_mps)
{
    if (!has_accepted_command)
        return 0.0;
    if (!(active_speed_ceiling_mps > 0.0) ||
        !std::isfinite(active_speed_ceiling_mps)) {
        return 0.0;
    }

    const double local_cap =
        std::isfinite(local_raceline_cap_mps)
        ? std::clamp(local_raceline_cap_mps, 0.0,
            active_speed_ceiling_mps)
        : active_speed_ceiling_mps;
    const double previous =
        std::isfinite(previous_command_mps)
        ? std::clamp(previous_command_mps, 0.0, local_cap)
        : 0.0;
    const double observed =
        std::isfinite(observed_speed_mps) && observed_speed_mps > 1.0e-6
        ? std::clamp(observed_speed_mps, 0.0, local_cap)
        : local_cap;
    const double requested =
        std::isfinite(requested_speed_mps) && requested_speed_mps > 1.0e-6
        ? std::clamp(requested_speed_mps, 0.0, local_cap)
        : local_cap;

    return std::clamp(
        std::min({previous, observed, requested, local_cap}),
        0.0, active_speed_ceiling_mps);
}

}  // namespace f1tenth_mpc

#endif

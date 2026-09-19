#include "mpc_authority_policy.hpp"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

void check(bool condition, const char *message)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void check_close(double actual, double expected, double tolerance,
                 const char *message)
{
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        std::fprintf(stderr, "FAIL: %s actual=%.9f expected=%.9f\n",
                     message, actual, expected);
        ++failures;
    }
}

}  // namespace

int main()
{
    using namespace f1tenth_mpc;

    check(mpc_localization_covariance_good(
              0.10, 0.20, 0.05, 0.25, 0.12),
          "PP-equivalent localization covariance accepts a good pose");
    check(!mpc_localization_covariance_good(
              0.30, 0.20, 0.05, 0.25, 0.12),
          "XY covariance outside the PP gate is rejected");
    check(!mpc_localization_covariance_good(
              0.10, 0.20, 0.13, 0.25, 0.12),
          "yaw covariance outside the PP gate is rejected");

    check_close(mpc_fallback_target_speed(
                    false, 2.0, 2.0, 2.0, 3.0, 3.0),
                0.0, 1.0e-12,
                "fallback cannot invent motion before first accepted MPC command");
    check_close(mpc_fallback_target_speed(
                    true, 2.2, 2.4, 2.3, 2.5, 3.0),
                2.2, 1.0e-12,
                "fallback never accelerates above the last accepted command");
    check_close(mpc_fallback_target_speed(
                    true, 2.8, 2.6, 2.7, 2.4, 3.0),
                2.4, 1.0e-12,
                "fallback respects the local raceline cap");

    check_close(mpc_select_current_steering(
                    0.20, true, 0.12, 0.5236),
                0.12, 1.0e-12,
                "fresh actuator feedback is the MPC steering state");
    check_close(mpc_select_current_steering(
                    0.20, false, -0.12, 0.5236),
                0.20, 1.0e-12,
                "stale actuator feedback leaves the modeled command state");
    check_close(mpc_select_current_steering(
                    0.20, true, 0.80, 0.5236),
                0.5236, 1.0e-12,
                "feedback is bounded by the physical steering envelope");

    if (failures != 0) {
        std::fprintf(stderr, "%d authority-policy test(s) failed\n", failures);
        return 1;
    }
    std::puts("mpc authority policy tests passed");
    return 0;
}

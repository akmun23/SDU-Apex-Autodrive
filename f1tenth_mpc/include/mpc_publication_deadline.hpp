#pragma once

#include <cmath>
#include <cstdint>
#include <utility>

namespace f1tenth_mpc
{

constexpr double kMpcHardPublicationBudgetMs = 20.0;

inline bool mpc_publication_budget_valid(double budget_ms) noexcept
{
  return std::isfinite(budget_ms) && budget_ms > 0.0 &&
         budget_ms <= kMpcHardPublicationBudgetMs;
}

// The publication gate is strict: exactly reaching the configured budget is
// a deadline miss, regardless of which already-validated plan supplied the
// command. Command-start lateness is measured separately by the caller.
inline bool mpc_cycle_within_publication_deadline(
  double elapsed_ms, double budget_ms) noexcept
{
  return std::isfinite(elapsed_ms) && std::isfinite(budget_ms) &&
         elapsed_ms >= 0.0 && mpc_publication_budget_valid(budget_ms) &&
         elapsed_ms < budget_ms;
}

// Keep the deadline check adjacent to the externally visible publication.
// The caller supplies a monotonic clock and the actual publish operation so
// this ordering is testable without ROS and cannot be bypassed by a separate
// candidate/recovery code path.
template<typename Clock, typename Publish>
inline bool mpc_publish_before_deadline(
  int64_t callback_start_ns,
  double budget_ms,
  double *elapsed_ms,
  Clock &&clock,
  Publish &&publish)
{
  const int64_t publish_start_ns = std::forward<Clock>(clock)();
  const double elapsed = static_cast<double>(
    publish_start_ns - callback_start_ns) * 1.0e-6;
  if (elapsed_ms) *elapsed_ms = elapsed;
  if (!mpc_cycle_within_publication_deadline(elapsed, budget_ms)) return false;
  std::forward<Publish>(publish)();
  return true;
}

}  // namespace f1tenth_mpc

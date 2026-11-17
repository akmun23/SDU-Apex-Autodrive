#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <utility>

namespace f1tenth_mpc
{

constexpr double kMpcHardPublicationBudgetMs = 22.0;

enum class MpcCommandEpochGate { Wait, Ready, Missed };

// Select a not-yet-issued stage of an independently validated plan.
// Wait for a full missed state period before advancing: normal inter-sample
// callbacks must leave the next fresh-state solve its original request window.
// Request stamps have sub-millisecond host jitter; round to the same command
// stages used by warm-start shifting, and never extend the terminal command.
inline int mpc_plan_continuation_stage(
  int64_t plan_ns, int64_t issued_ns, int64_t target_ns,
  int64_t period_ns, int stages, int64_t state_wait_ns) noexcept
{
  if (plan_ns <= 0 || issued_ns < plan_ns || target_ns <= issued_ns ||
      period_ns <= 0 || stages <= 0 || state_wait_ns < period_ns)
    return -1;
  const auto stage = [=](int64_t stamp) {
    const int64_t elapsed = stamp - plan_ns;
    return elapsed / period_ns + (elapsed % period_ns >= period_ns / 2 + period_ns % 2);
  };
  const int64_t next = stage(target_ns);
  return next > stage(issued_ns) && next < stages ? static_cast<int>(next) : -1;
}

inline double mpc_command_publication_deadline_us(
  double callback_deadline_us, double monotonic_now_us, int64_t ros_now_ns,
  int64_t intended_ns, int64_t transport_allowance_ns) noexcept
{
  return std::min(callback_deadline_us, monotonic_now_us +
    1.e-3 * (double(intended_ns - ros_now_ns) - double(transport_allowance_ns)));
}

inline bool mpc_next_bridge_request_epoch(
  int64_t source_ns, int64_t predicted_publish_ns, int64_t deadline_ns,
  int64_t period_ns, int64_t transport_allowance_ns, int64_t *epoch_ns) noexcept
{
  if (!epoch_ns || source_ns <= 0 || predicted_publish_ns < source_ns ||
      deadline_ns <= source_ns || period_ns <= 0 || transport_allowance_ns < 0)
    return false;
  const int64_t elapsed = predicted_publish_ns - source_ns;
  if (transport_allowance_ns > std::numeric_limits<int64_t>::max() - elapsed)
    return false;
  const int64_t predicted_previous_slot = (elapsed + transport_allowance_ns) / period_ns;
  // A later window is usable only if its preceding request can arrive before
  // the hard publication deadline. Otherwise choose the preceding window.
  const int64_t last_previous_slot = (deadline_ns - source_ns - 1) / period_ns;
  const int64_t slot = std::min(predicted_previous_slot, last_previous_slot) + 1;
  if (slot > (std::numeric_limits<int64_t>::max() - source_ns) / period_ns)
    return false;
  *epoch_ns = source_ns + slot * period_ns;
  return true;
}

// The bridge consumes the most recent command at each request. Publication
// must occur after the preceding request and leave the configured transport
// allowance before the intended request. Match request slots within half a
// period because actual request stamps contain host scheduling jitter.
inline MpcCommandEpochGate mpc_command_epoch_gate(
  int64_t intended_ns, int64_t last_request_ns, int64_t now_ns,
  int64_t period_ns, int64_t transport_allowance_ns) noexcept
{
  if (intended_ns <= 0 || last_request_ns <= 0 || now_ns < last_request_ns ||
      period_ns <= 0 || transport_allowance_ns < 0)
    return MpcCommandEpochGate::Missed;
  const int64_t remaining = intended_ns - last_request_ns;
  if (remaining <= period_ns / 2) return MpcCommandEpochGate::Missed;
  if (remaining - period_ns > period_ns / 2) return MpcCommandEpochGate::Wait;
  const int64_t elapsed = now_ns - last_request_ns;
  return elapsed < period_ns && transport_allowance_ns < period_ns - elapsed
    ? MpcCommandEpochGate::Ready : MpcCommandEpochGate::Missed;
}

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

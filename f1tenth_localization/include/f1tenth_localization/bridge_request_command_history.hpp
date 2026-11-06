#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace f1tenth_localization
{

// The odometry source plant must predict the exact actuator input captured
// by the Bridge at a request boundary, not a guessed callback-receipt epoch.
// Sensor headers and BridgeRequest commands carry the same integer nanosecond
// request stamp; NEVER match nearest or fall back to a different epoch here.
class BridgeRequestCommandHistory
{
public:
  static constexpr std::size_t kCapacity = 64;

  struct Command
  {
    int64_t stamp_ns{0};
    double steering_normalized{0.0};
    double throttle_normalized{0.0};
  };

  bool push(
    int64_t stamp_ns, double steering_normalized,
    double throttle_normalized) noexcept
  {
    if (stamp_ns <= 0 || !std::isfinite(steering_normalized) ||
      !std::isfinite(throttle_normalized) ||
      steering_normalized < -1.0 || steering_normalized > 1.0 ||
      throttle_normalized < -1.0 || throttle_normalized > 1.0)
    {
      return false;
    }
    samples_[next_] = {stamp_ns, steering_normalized, throttle_normalized};
    next_ = (next_ + 1U) % kCapacity;
    if (size_ < kCapacity) {
      ++size_;
    }
    return true;
  }

  bool find_exact(int64_t stamp_ns, Command * output) const noexcept
  {
    if (!output || stamp_ns <= 0) {
      return false;
    }
    const std::size_t first = (next_ + kCapacity - size_) % kCapacity;
    for (std::size_t offset = 0; offset < size_; ++offset) {
      const auto & candidate = samples_[(first + offset) % kCapacity];
      if (candidate.stamp_ns == stamp_ns) {
        *output = candidate;
        return true;
      }
    }
    return false;
  }

private:
  std::array<Command, kCapacity> samples_{};
  std::size_t next_{0};
  std::size_t size_{0};
};

}  // namespace f1tenth_localization

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace f1tenth_localization
{

// Small fixed-capacity history for headerless actuator-command topics.
// Sensor packets are stamped at the Bridge request epoch, so only command
// callbacks timestamped at or before that epoch may enter its delay queue.
class TimedCommandHistory
{
public:
  static constexpr std::size_t kCapacity = 8;

  void push(int64_t stamp_ns, double value) noexcept
  {
    samples_[next_] = {stamp_ns, value};
    next_ = (next_ + 1U) % kCapacity;
    if (size_ < kCapacity) {
      ++size_;
    }
  }

  double value_at_or_before(int64_t stamp_ns, double initial_value) const noexcept
  {
    bool found = false;
    int64_t best_stamp = 0;
    double best_value = initial_value;
    const std::size_t first = (next_ + kCapacity - size_) % kCapacity;
    for (std::size_t offset = 0; offset < size_; ++offset) {
      const auto & sample = samples_[(first + offset) % kCapacity];
      if (sample.stamp_ns <= stamp_ns &&
        (!found || sample.stamp_ns >= best_stamp))
      {
        found = true;
        best_stamp = sample.stamp_ns;
        best_value = sample.value;
      }
    }
    return best_value;
  }

private:
  struct Sample
  {
    int64_t stamp_ns{0};
    double value{0.0};
  };

  std::array<Sample, kCapacity> samples_{};
  std::size_t next_{0};
  std::size_t size_{0};
};

}  // namespace f1tenth_localization

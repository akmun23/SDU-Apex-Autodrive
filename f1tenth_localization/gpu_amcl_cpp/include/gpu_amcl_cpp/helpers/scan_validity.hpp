#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gpu_amcl_cpp::scan_validity {

struct SampledRangeCounts {
  std::size_t sampled = 0;
  std::size_t valid = 0;
};

// Match the sensor model's integer-stride beam subsampling exactly. A scan
// with no usable sampled return is odometry-only evidence, not a map update.
inline SampledRangeCounts count_sampled_valid_ranges(
    const std::vector<float>& ranges,
    int max_beams,
    double min_range,
    double max_range) {
  SampledRangeCounts counts;
  if (ranges.empty() || max_beams <= 0 || !std::isfinite(min_range) ||
      !std::isfinite(max_range) || min_range > max_range) {
    return counts;
  }

  const std::size_t step = std::max<std::size_t>(
      1, ranges.size() / static_cast<std::size_t>(max_beams));
  for (std::size_t index = 0; index < ranges.size(); index += step) {
    ++counts.sampled;
    const float range = ranges[index];
    if (std::isfinite(range) && range >= min_range && range <= max_range) {
      ++counts.valid;
    }
  }
  return counts;
}

}  // namespace gpu_amcl_cpp::scan_validity

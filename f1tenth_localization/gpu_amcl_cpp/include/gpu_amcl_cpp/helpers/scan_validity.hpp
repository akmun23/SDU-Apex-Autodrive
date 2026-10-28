#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace gpu_amcl_cpp::scan_validity {

struct SampledRangeCounts {
  std::size_t sampled = 0;
  std::size_t valid = 0;
};

// Header/receipt time does not make a held simulator range array a new
// observation. Compare the complete used measurement, not a subset/hash.
// Byte comparison also handles matching NaN payloads without fuzzy thresholds.
template<typename Scan>
bool same_observation(const Scan& a, const Scan& b) {
  return !a.ranges.empty() && a.ranges.size() == b.ranges.size() &&
      a.header.frame_id == b.header.frame_id &&
      a.angle_min == b.angle_min && a.angle_max == b.angle_max &&
      a.angle_increment == b.angle_increment &&
      a.range_min == b.range_min && a.range_max == b.range_max &&
      std::memcmp(a.ranges.data(), b.ranges.data(),
          a.ranges.size() * sizeof(float)) == 0;
}

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

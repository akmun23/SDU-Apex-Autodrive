#pragma once

#include <algorithm>
#include <iterator>
#include <utility>

namespace gpu_amcl_cpp::odom_history
{
// History must be strictly ordered by source stamp. A missing future endpoint
// means wait for odometry; substituting an older pose changes the scan's epoch.
template<typename History, typename Stamp>
auto bracket(const History & history, const Stamp & stamp)
  -> std::pair<const typename History::value_type *, const typename History::value_type *>
{
  if (history.empty() || stamp < history.front().stamp || stamp > history.back().stamp) {
    return {nullptr, nullptr};
  }
  const auto after = std::lower_bound(history.begin(), history.end(), stamp,
    [](const auto & sample, const auto & target) {return sample.stamp < target;});
  if (after->stamp == stamp) {
    return {&*after, &*after};
  }
  return {&*std::prev(after), &*after};
}
}  // namespace gpu_amcl_cpp::odom_history

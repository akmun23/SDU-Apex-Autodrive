#ifndef F1TENTH_MPC_REPLAY_METRICS_HPP
#define F1TENTH_MPC_REPLAY_METRICS_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace f1tenth_mpc::replay_metrics
{

// Nearest-rank percentile: sorted[ceil(p * N) - 1], with explicit extrema.
inline double percentile(std::vector<double> values, double fraction)
{
    if (values.empty() || !std::isfinite(fraction))
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    if (fraction <= 0.0) return values.front();
    if (fraction >= 1.0) return values.back();
    const auto rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, rank) - 1];
}

inline double minimum(const std::vector<double> &values)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    return *std::min_element(values.begin(), values.end());
}

inline double maximum(const std::vector<double> &values)
{
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    return *std::max_element(values.begin(), values.end());
}

}  // namespace f1tenth_mpc::replay_metrics

#endif  // F1TENTH_MPC_REPLAY_METRICS_HPP

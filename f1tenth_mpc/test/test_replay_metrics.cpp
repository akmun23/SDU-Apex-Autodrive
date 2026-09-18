#include "replay_metrics.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using f1tenth_mpc::replay_metrics::maximum;
using f1tenth_mpc::replay_metrics::minimum;
using f1tenth_mpc::replay_metrics::percentile;

int main()
{
    const std::vector<double> values{4.0, 1.0, 3.0, 2.0};
    if (percentile(values, 0.0) != 1.0) {
        std::fprintf(stderr, "percentile(0) must return the minimum\n");
        return 1;
    }
    if (percentile(values, 1.0) != 4.0) {
        std::fprintf(stderr, "percentile(1) must return the maximum\n");
        return 1;
    }
    if (percentile(values, 0.5) != 2.0) {
        std::fprintf(stderr, "interior percentile must use nearest-rank\n");
        return 1;
    }
    if (percentile(std::vector<double>{7.5}, 0.95) != 7.5) {
        std::fprintf(stderr, "one-element percentile must return that value\n");
        return 1;
    }
    if (!std::isnan(percentile(std::vector<double>{}, 0.5))) {
        std::fprintf(stderr, "empty percentile must return NaN\n");
        return 1;
    }
    if (minimum(values) != 1.0 || maximum(values) != 4.0 ||
        !std::isnan(minimum(std::vector<double>{}))) {
        std::fprintf(stderr, "direct extrema helpers returned wrong results\n");
        return 1;
    }
    return 0;
}

#include "common/math_utils.hpp"
#include <algorithm>


namespace f1tenth_control {
namespace math {

std::vector<double> medianFilter(const std::vector<double>& data, size_t window_size) {
    if (data.empty()) return {};
    if (window_size <= 1) return data;
    
    // Median requires an odd window; round up to preserve filter symmetry.
    window_size |= 1;
    
    const size_t half_window = window_size / 2;
    const size_t n = data.size();
    std::vector<double> result(n);
    std::vector<double> window(window_size);
    
    for (size_t i = 0; i < n; ++i) {
        const size_t start = (i >= half_window) ? i - half_window : 0;
        const size_t end = std::min(i + half_window + 1, n);
        const size_t win_len = end - start;
        
        std::copy(data.begin() + start, data.begin() + end, window.begin());
        
        const size_t mid = win_len / 2;
        std::nth_element(window.begin(), window.begin() + mid, window.begin() + win_len);
        result[i] = window[mid];
    }
    
    return result;
}

}  // namespace math
}  // namespace f1tenth_control

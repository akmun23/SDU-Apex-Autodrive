/** Normalize log weights on the GPU: subtract the maximum, multiply prior weights
 * by exp(log_w - max), then divide by their sum using CUB reductions. */

#include "gpu_amcl_cpp/helpers/cuda_utils.hpp"
#include <cub/cub.cuh>
#include <cmath>
#include <cfloat>

namespace gpu_amcl_cpp {


/// Fused exp-shift-multiply: w[i] = exp(log_w[i] - max_lw) * old_w[i]
__global__
void kernel_exp_shift_mul(const float* __restrict__ log_w,
                          const float* __restrict__ old_w,
                          float* __restrict__ out_w,
                          const float* __restrict__ d_max,
                          int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_w[i] = expf(log_w[i] - *d_max) * old_w[i];
}

/// Normalize in-place: w[i] /= sum.  Falls back to uniform if sum ≤ 0.
__global__
void kernel_normalize(float* __restrict__ w,
                      const float* __restrict__ d_sum,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s = *d_sum;
    if (s > 0.0f) {
        w[i] *= __frcp_rn(s);
    } else {
        w[i] = 1.0f / static_cast<float>(n);
    }
}


size_t query_cub_normalize_temp_bytes(int max_n) {
    size_t temp_max = 0, temp_sum = 0;
    float* dummy = nullptr;

    cub::DeviceReduce::Max(nullptr, temp_max, dummy, dummy, max_n);
    cub::DeviceReduce::Sum(nullptr, temp_sum, dummy, dummy, max_n);

    return std::max(temp_max, temp_sum);
}


void launch_gpu_normalize_weights(
        const float* d_log_w,
        float* d_old_w,          // existing weights (read, then overwritten)
        float* d_scratch_w,      // scratch buffer (receives exp-shift-mul result)
        float* d_max_val,        // device scalar for CUB max output
        float* d_sum_val,        // device scalar for CUB sum output
        void* d_cub_temp,        // CUB temp storage
        size_t cub_temp_bytes,
        int n,
        cudaStream_t stream) {
    if (n <= 0) return;

    const auto launch = make_adaptive_launch_config(n);

    CUDA_CHECK(cub::DeviceReduce::Max(d_cub_temp, cub_temp_bytes,
                                      d_log_w, d_max_val, n, stream));

    kernel_exp_shift_mul<<<launch.grid, launch.block, 0, stream>>>(
        d_log_w, d_old_w, d_scratch_w, d_max_val, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes,
                                      d_scratch_w, d_sum_val, n, stream));

    kernel_normalize<<<launch.grid, launch.block, 0, stream>>>(d_scratch_w, d_sum_val, n);
    CUDA_CHECK(cudaGetLastError());

    // Normalized weights reside in d_scratch_w.
    // Caller is responsible for swapping d_scratch_w ↔ d_old_w pointers
    // (or copying, but swap is preferred for zero overhead).
}

}  // namespace gpu_amcl_cpp

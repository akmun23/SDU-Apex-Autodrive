#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <memory>

namespace gpu_amcl_cpp {

/** Log the call site and CUDA error, then throw `std::runtime_error` on failure. */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            throw std::runtime_error(cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

struct CudaLaunchConfig {
    int grid = 0;
    int block = 256;
};

inline int ceil_div_int(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

inline int cuda_sm_count() {
    static int cached_sm_count = 0;
    if (cached_sm_count <= 0) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &cached_sm_count, cudaDevAttrMultiProcessorCount, device));
        if (cached_sm_count <= 0) {
            cached_sm_count = 1;
        }
    }
    return cached_sm_count;
}

inline CudaLaunchConfig make_adaptive_launch_config(
        int work_items,
        int max_block_threads = 256,
        int min_block_threads = 32,
        int target_blocks = 0) {
    if (work_items <= 0) {
        return CudaLaunchConfig{0, max_block_threads};
    }

    const int target = target_blocks > 0 ? target_blocks : cuda_sm_count();
    int block = max_block_threads;
    while (block > min_block_threads && ceil_div_int(work_items, block) < target) {
        block /= 2;
    }

    return CudaLaunchConfig{ceil_div_int(work_items, block), block};
}

/** Own a CUDA allocation; `upload()` and `download()` transfer host/device data. */
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) { 
        allocate(count); 
    }
    ~DeviceBuffer() { free(); }

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_   = nullptr;
        o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_     = o.ptr_;
            count_   = o.count_;
            o.ptr_   = nullptr;
            o.count_ = 0;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /** Release the allocation before allocating `count` elements; throw on CUDA failure. */
    void allocate(size_t count) {
        free();
        if (count > (std::numeric_limits<size_t>::max() / sizeof(T))) {
            throw std::overflow_error("DeviceBuffer allocation size overflow");
        }

        T* new_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_ptr), count * sizeof(T)));

        ptr_ = new_ptr;
        count_ = count;
    }

    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_   = nullptr;
            count_ = 0;
        }
    }

    /** Copy `count` host elements, growing the allocation if needed; throw on CUDA failure. */
    void upload(const T* host_data, size_t count) {
        if (count > count_) allocate(count);
        CUDA_CHECK(cudaMemcpy(ptr_, host_data,
                              count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    /** Copy `count` elements to host memory; throw on CUDA failure. */
    void download(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_,
                              count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    /** Queue a host-to-device copy on `stream`, growing the buffer if needed.
     * Throw on CUDA failure. */
    void upload_async(const T* host_data, size_t count,
                      cudaStream_t stream) {
        if (count > count_) allocate(count);
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host_data,
                                   count * sizeof(T),
                                   cudaMemcpyHostToDevice, stream));
    }

    /** Queue a device-to-host copy on `stream`; host data is valid after stream completion.
     * Throw on CUDA failure. */
    void download_async(T* host_data, size_t count,
                        cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(host_data, ptr_,
                                   count * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream));
    }

    void zero() {
        if (ptr_)
            CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

    T*     ptr()   const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};

class CudaTimingEvent {
public:
    CudaTimingEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaTimingEvent() { if (event_) cudaEventDestroy(event_); }
    CudaTimingEvent(const CudaTimingEvent&) = delete;
    CudaTimingEvent& operator=(const CudaTimingEvent&) = delete;
    void record(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(event_, stream)); }
    void wait() const { CUDA_CHECK(cudaEventSynchronize(event_)); }
    // Both events must have completed at an existing host-read boundary.
    double elapsed_ms_since(const CudaTimingEvent& start) const {
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.event_, event_));
        return elapsed;
    }
private:
    cudaEvent_t event_ = nullptr;
};

class CudaStream {
public:
    CudaStream(){ 
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA devices available");
        }
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(0));
        cudaError_t stream_status = cudaStreamCreate(&s_);
        if (stream_status == cudaErrorNotSupported) {
            s_ = 0;
            std::fprintf(stderr,
                         "[gpu_amcl_cpp] CUDA stream creation not supported; using default stream\n");
            return;
        }
        CUDA_CHECK(stream_status);
    }
    ~CudaStream(){ 
        if (s_) cudaStreamDestroy(s_); 
    }

    CudaStream(CudaStream&& o) noexcept : s_(o.s_){
        o.s_ = nullptr; 
    }
    CudaStream& operator=(CudaStream&& o) noexcept {
        if (this != &o) {
            if (s_) cudaStreamDestroy(s_);
            s_   = o.s_;
            o.s_ = nullptr;
        }
        return *this;
    }
    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const{ 
        return s_; 
    }

    /** Wait for queued stream operations; throw on CUDA failure. */
    void sync() const{
        CUDA_CHECK(cudaStreamSynchronize(s_)); 
    }

private:
    cudaStream_t s_ = nullptr;
};



inline int grid_size(int n, int block_size = 256) {
    return (n + block_size - 1) / block_size;
}

}  // namespace gpu_amcl_cpp

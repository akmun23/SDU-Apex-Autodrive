#pragma once

// The ROS telemetry queue and timing recorder are deliberately separate from
// the time-critical MPC controller. A pending telemetry frame may be replaced;
// it never blocks the control decision on JSON serialization or ROS publish.
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <time.h>
#include <utility>

namespace f1tenth_mpc {

inline int64_t steady_time_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int64_t thread_cpu_time_ns()
{
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return 0;
    return static_cast<int64_t>(value.tv_sec) * 1'000'000'000LL +
        value.tv_nsec;
}

struct DiagnosticTimingSample
{
    bool valid{};
    double wall_us{};
    double thread_cpu_us{};
    double plant_wall_us{};
    double plant_thread_cpu_us{};
    double json_build_wall_us{};
    double json_build_thread_cpu_us{};
    double queue_enqueue_wall_us{};
    double queue_enqueue_thread_cpu_us{};
    int reconstructed_plant_steps{};
};

class DiagnosticTimingRecorder
{
public:
    explicit DiagnosticTimingRecorder(DiagnosticTimingSample &sample)
        : sample_(sample), wall_start_ns_(steady_time_ns()),
          cpu_start_ns_(thread_cpu_time_ns())
    {
        sample_ = {};
    }

    void begin_plant()
    {
        phase_wall_start_ns_ = steady_time_ns();
        phase_cpu_start_ns_ = thread_cpu_time_ns();
    }

    void end_plant(int reconstructed_steps)
    {
        sample_.plant_wall_us = elapsed_us(phase_wall_start_ns_, steady_time_ns());
        sample_.plant_thread_cpu_us =
            elapsed_us(phase_cpu_start_ns_, thread_cpu_time_ns());
        sample_.reconstructed_plant_steps = reconstructed_steps;
    }

    void begin_json_build()
    {
        phase_wall_start_ns_ = steady_time_ns();
        phase_cpu_start_ns_ = thread_cpu_time_ns();
    }

    void end_json_build()
    {
        sample_.json_build_wall_us =
            elapsed_us(phase_wall_start_ns_, steady_time_ns());
        sample_.json_build_thread_cpu_us =
            elapsed_us(phase_cpu_start_ns_, thread_cpu_time_ns());
    }

    void begin_enqueue()
    {
        phase_wall_start_ns_ = steady_time_ns();
        phase_cpu_start_ns_ = thread_cpu_time_ns();
    }

    void end_enqueue()
    {
        sample_.queue_enqueue_wall_us =
            elapsed_us(phase_wall_start_ns_, steady_time_ns());
        sample_.queue_enqueue_thread_cpu_us =
            elapsed_us(phase_cpu_start_ns_, thread_cpu_time_ns());
    }

    ~DiagnosticTimingRecorder()
    {
        sample_.wall_us = elapsed_us(wall_start_ns_, steady_time_ns());
        sample_.thread_cpu_us = elapsed_us(cpu_start_ns_, thread_cpu_time_ns());
        sample_.valid = true;
    }

private:
    static double elapsed_us(int64_t start_ns, int64_t finish_ns)
    {
        return start_ns > 0 && finish_ns >= start_ns
            ? static_cast<double>(finish_ns - start_ns) * 1.0e-3 : 0.0;
    }

    DiagnosticTimingSample &sample_;
    int64_t wall_start_ns_{};
    int64_t cpu_start_ns_{};
    int64_t phase_wall_start_ns_{};
    int64_t phase_cpu_start_ns_{};
};

class DiagnosticPublisherWorker
{
public:
    struct Stats
    {
        std::uint64_t published_frames{};
        std::uint64_t replaced_frames{};
        std::uint64_t publish_failures{};
        double last_queue_wait_us{};
        double last_serialization_wall_us{};
        double last_serialization_thread_cpu_us{};
        double last_publish_wall_us{};
        double last_publish_thread_cpu_us{};
        bool lower_priority{};
    };

    explicit DiagnosticPublisherWorker(
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher)
        : publisher_(std::move(publisher)), worker_([this] { run(); })
    {
    }

    ~DiagnosticPublisherWorker()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    bool enqueue(std::ostringstream && json_stream)
    {
        Frame frame;
        frame.has_json_stream = true;
        frame.json_stream = std::move(json_stream);
        frame.enqueue_steady_ns = steady_time_ns();
        return enqueue_frame(std::move(frame));
    }

    bool enqueue(std::string serialized_message)
    {
        Frame frame;
        frame.serialized_message = std::move(serialized_message);
        frame.enqueue_steady_ns = steady_time_ns();
        return enqueue_frame(std::move(frame));
    }

    Stats stats() const
    {
        return {
            published_frames_.load(std::memory_order_relaxed),
            replaced_frames_.load(std::memory_order_relaxed),
            publish_failures_.load(std::memory_order_relaxed),
            last_queue_wait_us_.load(std::memory_order_relaxed),
            last_serialization_wall_us_.load(std::memory_order_relaxed),
            last_serialization_thread_cpu_us_.load(std::memory_order_relaxed),
            last_publish_wall_us_.load(std::memory_order_relaxed),
            last_publish_thread_cpu_us_.load(std::memory_order_relaxed),
            lower_priority_.load(std::memory_order_relaxed),
        };
    }

private:
    struct Frame
    {
        bool has_json_stream{};
        std::string serialized_message;
        std::ostringstream json_stream;
        int64_t enqueue_steady_ns{};
    };

    bool enqueue_frame(Frame frame)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return false;
            if (pending_) replaced_frames_.fetch_add(1,
                std::memory_order_relaxed);
            pending_.emplace(std::move(frame));
        }
        condition_.notify_one();
        return true;
    }

    void run();

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<Frame> pending_;
    bool stopping_{};
    std::thread worker_;
    std::atomic<std::uint64_t> published_frames_{};
    std::atomic<std::uint64_t> replaced_frames_{};
    std::atomic<std::uint64_t> publish_failures_{};
    std::atomic<double> last_queue_wait_us_{};
    std::atomic<double> last_serialization_wall_us_{};
    std::atomic<double> last_serialization_thread_cpu_us_{};
    std::atomic<double> last_publish_wall_us_{};
    std::atomic<double> last_publish_thread_cpu_us_{};
    std::atomic<bool> lower_priority_{};
};


} // namespace f1tenth_mpc

// Asynchronous, lower-priority ROS telemetry worker.
// This cannot block the 20ms control callback on serialization/publish.
#include "mpc_diagnostic_worker.hpp"

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace f1tenth_mpc {

void DiagnosticPublisherWorker::run()
{
#if defined(__linux__)
    const auto thread_id = static_cast<id_t>(syscall(SYS_gettid));
    lower_priority_.store(
        setpriority(PRIO_PROCESS, thread_id, 5) == 0,
        std::memory_order_relaxed);
#endif
    for (;;) {
        Frame frame;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || pending_; });
            if (!pending_ && stopping_) return;
            frame = std::move(*pending_);
            pending_.reset();
        }

        const int64_t dequeued_ns = steady_time_ns();
        last_queue_wait_us_.store(
            frame.enqueue_steady_ns > 0 && dequeued_ns >= frame.enqueue_steady_ns
                ? static_cast<double>(dequeued_ns - frame.enqueue_steady_ns) * 1.0e-3
                : 0.0,
            std::memory_order_relaxed);

        std_msgs::msg::String message;
        try {
            if (frame.has_json_stream) {
                const int64_t serialize_wall_start_ns = steady_time_ns();
                const int64_t serialize_cpu_start_ns = thread_cpu_time_ns();
                message.data = frame.json_stream.str();
                const int64_t serialize_wall_end_ns = steady_time_ns();
                const int64_t serialize_cpu_end_ns = thread_cpu_time_ns();
                last_serialization_wall_us_.store(
                    static_cast<double>(serialize_wall_end_ns -
                        serialize_wall_start_ns) * 1.0e-3,
                    std::memory_order_relaxed);
                last_serialization_thread_cpu_us_.store(
                    serialize_cpu_start_ns > 0 && serialize_cpu_end_ns >=
                            serialize_cpu_start_ns
                        ? static_cast<double>(serialize_cpu_end_ns -
                            serialize_cpu_start_ns) * 1.0e-3
                        : 0.0,
                    std::memory_order_relaxed);
            } else {
                message.data = std::move(frame.serialized_message);
                last_serialization_wall_us_.store(0.0,
                    std::memory_order_relaxed);
                last_serialization_thread_cpu_us_.store(0.0,
                    std::memory_order_relaxed);
            }
        } catch (...) {
            publish_failures_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const int64_t publish_wall_start_ns = steady_time_ns();
        const int64_t publish_cpu_start_ns = thread_cpu_time_ns();
        try {
            publisher_->publish(message);
            published_frames_.fetch_add(1, std::memory_order_relaxed);
        } catch (...) {
            publish_failures_.fetch_add(1, std::memory_order_relaxed);
        }
        const int64_t publish_wall_end_ns = steady_time_ns();
        const int64_t publish_cpu_end_ns = thread_cpu_time_ns();
        last_publish_wall_us_.store(
            static_cast<double>(publish_wall_end_ns -
                publish_wall_start_ns) * 1.0e-3,
            std::memory_order_relaxed);
        last_publish_thread_cpu_us_.store(
            publish_cpu_start_ns > 0 && publish_cpu_end_ns >=
                    publish_cpu_start_ns
                ? static_cast<double>(publish_cpu_end_ns -
                    publish_cpu_start_ns) * 1.0e-3
                : 0.0,
            std::memory_order_relaxed);
    }
}

} // namespace f1tenth_mpc

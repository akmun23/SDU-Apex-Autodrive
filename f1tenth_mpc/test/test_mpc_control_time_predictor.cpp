#include "mpc_control_time_predictor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using namespace f1tenth_mpc;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "check failed at " << __FILE__ << ':' << __LINE__ \
                  << ": " #condition << '\n'; \
        std::abort(); \
    } \
} while (0)

constexpr double kPi = 3.14159265358979323846;
constexpr int64_t kSourceStampNs = 1000000000LL;

double wrap_angle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

std::vector<MpcTrajectorySample_t> make_circle(double radius)
{
    constexpr std::size_t kPoints = 720;
    std::vector<MpcTrajectorySample_t> points(kPoints);
    const double circumference = 2.0 * kPi * radius;
    for (std::size_t i = 0; i < kPoints; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) /
            static_cast<double>(kPoints);
        MpcTrajectorySample_t point{};
        point.s = circumference * static_cast<double>(i) /
            static_cast<double>(kPoints);
        point.x = radius * std::cos(angle);
        point.y = radius * std::sin(angle);
        point.heading = angle + 0.5 * kPi;
        point.curvature = 1.0 / radius;
        point.speed = 2.0;
        point.left_bound = 1.0;
        point.right_bound = 1.0;
        points[i] = point;
    }
    std::size_t count = points.size();
    double lap_length = 0.0;
    CHECK(mpc_trajectory_prepare(points.data(), &count, &lap_length));
    CHECK(lap_length > 0.0);
    points.resize(count);
    return points;
}

MpcSynchronizedState make_source()
{
    MpcSynchronizedState source{};
    source.source_stamp_ns = kSourceStampNs;
    source.map_x = 10.0;
    source.map_y = 0.0;
    source.map_yaw = 0.5 * kPi;
    source.u = 2.0;
    source.v = 0.0;
    return source;
}

MpcControlTimePrediction predict(
    MpcControlTimeMode mode, const MpcSynchronizedState &source,
    int64_t target_stamp_ns, const MpcCommandHistory &history,
    const std::vector<MpcTrajectorySample_t> &trajectory, double lap_length)
{
    MpcControlTimePrediction result{};
    MpcControlTimePredictorConfig config;
    CHECK(predict_to_control_time(mode, source, target_stamp_ns, history,
        trajectory.data(), trajectory.size(), lap_length,
        std::numeric_limits<std::size_t>::max(), nullptr, config, &result) ==
        MpcControlTimeStatus::kOk);
    return result;
}

void test_constant_twist_25_50_100_ms()
{
    const auto trajectory = make_circle(10.0);
    const double lap_length = 2.0 * kPi * 10.0;
    const MpcSynchronizedState source = make_source();
    MpcCommandHistory empty_history;
    for (const int64_t age_ms : {25, 50, 100}) {
        const auto result = predict(MpcControlTimeMode::kConstantBodyTwist,
            source, kSourceStampNs + age_ms * 1000000LL, empty_history,
            trajectory, lap_length);
        CHECK(std::abs(result.state.map_x - 10.0) < 1.0e-8);
        CHECK(std::abs(result.state.map_y - 2.0 * age_ms / 1000.0) < 1.0e-7);
        CHECK(std::abs(result.age_s - age_ms / 1000.0) < 1.0e-12);
    }
}

void test_constant_twist_yaw_and_symmetry()
{
    const auto trajectory = make_circle(10.0);
    const double lap_length = 2.0 * kPi * 10.0;
    MpcCommandHistory empty_history;
    auto source = make_source();
    source.yaw_rate = 0.5;
    const double dt = 0.1;
    const double midpoint_yaw = source.map_yaw + 0.5 * source.yaw_rate * dt;
    const auto turning = predict(MpcControlTimeMode::kConstantBodyTwist,
        source, kSourceStampNs + 100000000LL, empty_history, trajectory,
        lap_length);
    CHECK(std::abs(turning.state.map_x -
        (source.map_x + dt * source.u * std::cos(midpoint_yaw))) < 1.0e-8);
    CHECK(std::abs(turning.state.map_y -
        (source.map_y + dt * source.u * std::sin(midpoint_yaw))) < 1.0e-8);
    CHECK(std::abs(std::atan2(std::sin(turning.state.map_yaw -
        (source.map_yaw + 0.05)), std::cos(turning.state.map_yaw -
        (source.map_yaw + 0.05)))) < 1.0e-10);

    source.yaw_rate = 0.0;
    source.v = 0.5;
    const auto left = predict(MpcControlTimeMode::kConstantBodyTwist, source,
        kSourceStampNs + 50000000LL, empty_history, trajectory, lap_length);
    source.v = -0.5;
    const auto right = predict(MpcControlTimeMode::kConstantBodyTwist, source,
        kSourceStampNs + 50000000LL, empty_history, trajectory, lap_length);
    CHECK(std::abs(left.state.map_x + right.state.map_x - 20.0) < 1.0e-8);
    CHECK(std::abs(left.state.map_y - right.state.map_y) < 1.0e-8);

    source.map_x = 0.0;
    source.map_y = 10.0;
    source.map_yaw = kPi - 0.01;
    source.v = 0.0;
    source.yaw_rate = 0.5;
    const auto wrapped = predict(MpcControlTimeMode::kConstantBodyTwist,
        source, kSourceStampNs + 50000000LL, empty_history, trajectory,
        lap_length);
    CHECK(wrapped.state.map_yaw >= -kPi && wrapped.state.map_yaw <= kPi);
    CHECK(std::abs(wrap_angle(wrapped.state.map_yaw - (-kPi + 0.015))) <
        1.0e-8);
}

void test_ct0_and_timing_fallback()
{
    const auto trajectory = make_circle(10.0);
    const double lap_length = 2.0 * kPi * 10.0;
    const MpcSynchronizedState source = make_source();
    MpcCommandHistory history;
    const auto unchanged = predict(MpcControlTimeMode::kNoExtrapolation,
        source, kSourceStampNs + 50000000LL, history, trajectory, lap_length);
    CHECK(unchanged.state.map_x == source.map_x);
    CHECK(unchanged.state.map_y == source.map_y);
    CHECK(unchanged.state.map_yaw == source.map_yaw);

    MpcControlTimePrediction fallback{};
    MpcControlTimePredictorConfig config;
    CHECK(predict_to_control_time(MpcControlTimeMode::kNoExtrapolation,
        source, kSourceStampNs - 1, history, trajectory.data(),
        trajectory.size(), lap_length,
        std::numeric_limits<std::size_t>::max(), nullptr, config, &fallback) ==
        MpcControlTimeStatus::kOk);
    CHECK(fallback.used_time_fallback);
    CHECK(fallback.state.map_x == source.map_x);
    CHECK(predict_to_control_time(MpcControlTimeMode::kNoExtrapolation,
        source, kSourceStampNs + 120000001LL, history, trajectory.data(),
        trajectory.size(), lap_length,
        std::numeric_limits<std::size_t>::max(), nullptr, config, &fallback) ==
        MpcControlTimeStatus::kOk);
    CHECK(fallback.used_time_fallback);
    CHECK(fallback.age_s > config.maximum_state_age_s);
}

void test_history_order_wrap_and_future_causality()
{
    MpcCommandHistory history;
    CHECK(history.push({kSourceStampNs - 100000000LL, 0.0, 2.0}));
    CHECK(history.push({kSourceStampNs - 100000000LL, 0.1, 3.0}));
    for (int i = 1; i <= 20; ++i) {
        CHECK(history.push({kSourceStampNs + i * 1000000LL,
            0.001 * i, 2.0 + 0.1 * i}));
    }
    CHECK(history.size() == f1tenth_mpc::kMpcCommandHistoryCapacity);
    MpcCommandHistoryEntry oldest{};
    CHECK(history.at(0, &oldest));
    CHECK(oldest.stamp_ns == kSourceStampNs + 5000000LL);

    const auto trajectory = make_circle(10.0);
    const double lap_length = 2.0 * kPi * 10.0;
    MpcCommandHistory causal_history;
    CHECK(causal_history.push({kSourceStampNs - 100000000LL, 0.0, 2.0}));
    const auto without_future = predict(
        MpcControlTimeMode::kAcceptedModelCommandHistory, make_source(),
        kSourceStampNs + 25000000LL, causal_history, trajectory, lap_length);
    CHECK(causal_history.push({kSourceStampNs + 50000000LL, 0.4, 8.0}));
    const auto with_future = predict(
        MpcControlTimeMode::kAcceptedModelCommandHistory, make_source(),
        kSourceStampNs + 25000000LL, causal_history, trajectory, lap_length);
    CHECK(with_future.state.map_x == without_future.state.map_x);
    CHECK(with_future.state.map_y == without_future.state.map_y);
    CHECK(with_future.state.map_yaw == without_future.state.map_yaw);
    CHECK(with_future.state.u == without_future.state.u);
    CHECK(with_future.target_speed_mps == without_future.target_speed_mps);
    CHECK(with_future.previous_steering_rate_radps ==
        without_future.previous_steering_rate_radps);
    CHECK(with_future.previous_target_speed_rate_mps2 ==
        without_future.previous_target_speed_rate_mps2);
    CHECK(with_future.command_changes_used == 0);
}

void test_model_prediction_and_zero_history_fallback()
{
    const auto trajectory = make_circle(10.0);
    const double lap_length = 2.0 * kPi * 10.0;
    MpcCommandHistory history;
    CHECK(history.push({kSourceStampNs - 100000000LL, 0.0, 2.0}));
    CHECK(history.push({kSourceStampNs + 25000000LL, 0.1, 4.0}));
    const auto command_change = predict(
        MpcControlTimeMode::kAcceptedModelCommandHistory, make_source(),
        kSourceStampNs + 50000000LL, history, trajectory, lap_length);
    CHECK(command_change.command_changes_used == 1);
    CHECK(command_change.state.u > 2.0);
    CHECK(command_change.progress_m > command_change.projection.s - 1.0e-6);

    MpcCommandHistory empty;
    const auto fallback_a = predict(
        MpcControlTimeMode::kAcceptedModelCommandHistory, make_source(),
        kSourceStampNs + 100000000LL, empty, trajectory, lap_length);
    const auto fallback_b = predict(
        MpcControlTimeMode::kAcceptedModelCommandHistory, make_source(),
        kSourceStampNs + 100000000LL, empty, trajectory, lap_length);
    CHECK(fallback_a.used_command_fallback);
    CHECK(fallback_a.state.map_x == fallback_b.state.map_x);
    CHECK(fallback_a.state.map_y == fallback_b.state.map_y);
    CHECK(fallback_a.state.u == fallback_b.state.u);
}

}  // namespace

int main()
{
    test_constant_twist_25_50_100_ms();
    test_constant_twist_yaw_and_symmetry();
    test_ct0_and_timing_fallback();
    test_history_order_wrap_and_future_causality();
    test_model_prediction_and_zero_history_fallback();
    std::cout << "MPC control-time predictor tests passed\n";
    return 0;
}

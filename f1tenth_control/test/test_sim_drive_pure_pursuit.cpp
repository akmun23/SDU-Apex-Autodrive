#include "algorithms/pure_pursuit.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using f1tenth_control::Point2D;
using f1tenth_control::Pose2D;
using f1tenth_control::PurePursuit;
using f1tenth_control::PurePursuitConfig;
using f1tenth_control::PurePursuitOutput;
using f1tenth_control::TrajectoryPoint;
using f1tenth_control::VehicleState;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDt = 0.1;
constexpr double kWheelbase = 0.324;
constexpr double kMaxSteering = 0.5236;
constexpr double kMaxSteeringRate = 3.2;

TrajectoryPoint point(
    double s, double x, double y, double heading, double curvature,
    double speed, double left_bound = 2.0, double right_bound = 2.0) {
    TrajectoryPoint result;
    result.arc_length = s;
    result.x = x;
    result.y = y;
    result.heading = heading;
    result.curvature = curvature;
    result.velocity = speed;
    result.left_bound = left_bound;
    result.right_bound = right_bound;
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void assert_finite_command(const f1tenth_control::PurePursuitOutput& output) {
    require(output.valid, "Pure Pursuit returned an invalid command");
    require(std::isfinite(output.steering_angle), "steering is not finite");
    require(std::isfinite(output.target_speed), "speed is not finite");
    require(std::isfinite(output.cross_track_error), "cross-track error is not finite");
    require(std::abs(output.steering_angle) <= 0.5236 + 1.0e-9,
            "steering exceeded the documented limit");
}

void test_five_point_five_metre_per_second_straight() {
    PurePursuitConfig config;
    config.min_lookahead = 0.65;
    config.max_lookahead = 1.15;
    config.lookahead_gain = 0.14;
    config.max_lateral_accel = 6.5;
    config.wall_bias_gain = 0.0;

    std::vector<TrajectoryPoint> path;
    for (int i = 0; i <= 800; ++i) {
        const double x = 0.05 * static_cast<double>(i);
        path.push_back(point(x, x, 0.0, 0.0, 0.0, 5.5));
    }

    PurePursuit controller(config);
    controller.setTrajectory(path);

    VehicleState state;
    state.pose = Pose2D(1.0, 0.0, 0.0);
    state.velocity = 5.5;
    const auto output = controller.compute(state);
    assert_finite_command(output);
    // The controller core must not silently clamp a valid high-speed straight
    // to the old 3.5 m/s practice profile. The runtime node applies its own
    // explicit max_speed parameter at the command boundary.
    require(output.target_speed > 5.0, "high-speed straight was clamped below 5 m/s");
    require(std::abs(output.steering_angle) < 1.0e-6,
            "straight-line steering was not zero");
}

void test_ten_hz_straight_closed_loop() {
    PurePursuitConfig config;
    config.wall_bias_gain = 0.0;
    config.max_lateral_accel = 6.5;

    std::vector<TrajectoryPoint> path;
    for (int i = 0; i <= 1000; ++i) {
        const double x = 0.05 * static_cast<double>(i);
        path.push_back(point(x, x, 0.0, 0.0, 0.0, 5.5));
    }

    PurePursuit controller(config);
    controller.setTrajectory(path);
    VehicleState state;
    state.velocity = 5.5;

    // One compute event per official scan.  The test intentionally uses 100
    // ms spacing; no timer-rate assumption is hidden in the algorithm core.
    for (int step = 0; step < 80; ++step) {
        state.pose.x = 0.5 + 0.55 * static_cast<double>(step);
        const auto output = controller.compute(state);
        assert_finite_command(output);
        require(output.target_speed > 5.0,
                "10 Hz straight target was clamped below 5 m/s");
        require(std::abs(output.cross_track_error) < 1.0e-9,
                "10 Hz straight accumulated cross-track error");
    }
}

void test_positive_curvature_turn() {
    PurePursuitConfig config;
    config.wall_bias_gain = 0.0;
    config.curvature_feedforward_gain = 0.25;

    constexpr double radius = 3.0;
    constexpr int samples = 240;
    std::vector<TrajectoryPoint> path;
    for (int i = 0; i < samples; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) / samples;
        const double heading = angle + 0.5 * kPi;
        path.push_back(point(
            radius * angle, radius * std::cos(angle), radius * std::sin(angle),
            heading, 1.0 / radius, 3.5));
    }

    PurePursuit controller(config);
    controller.setTrajectory(path);
    VehicleState state;
    state.pose = Pose2D(radius, 0.0, 0.5 * kPi);
    state.velocity = 3.5;
    const auto output = controller.compute(state);
    assert_finite_command(output);
    require(output.steering_angle > 0.0, "positive-curvature steering sign is wrong");
    require(output.target_speed <= 3.5 + 1.0e-9,
            "corner speed exceeded the lateral-acceleration profile");
}

void test_preview_brakes_before_slow_section() {
    PurePursuitConfig config;
    config.wall_bias_gain = 0.0;
    config.speed_preview_distance = 4.0;
    config.speed_profile_braking_decel = 1.5;

    std::vector<TrajectoryPoint> path;
    for (int i = 0; i <= 100; ++i) {
        const double x = 0.1 * static_cast<double>(i);
        const double speed = x < 3.0 ? 5.5 : 2.0;
        path.push_back(point(x, x, 0.0, 0.0, 0.0, speed));
    }

    PurePursuit controller(config);
    controller.setTrajectory(path);
    VehicleState state;
    state.pose = Pose2D(0.0, 0.0, 0.0);
    state.velocity = 5.5;
    const auto output = controller.compute(state);
    assert_finite_command(output);
    require(output.target_speed < 4.5,
            "preview braking did not start before the slow section");
    require(output.target_speed > 0.12,
            "preview braking collapsed to a stop on a valid path");
}

void test_narrow_corridor_bias_is_bounded() {
    PurePursuitConfig config;
    config.wall_bias_gain = 1.0;
    config.wall_bias_max_m = 0.10;

    std::vector<TrajectoryPoint> path;
    for (int i = 0; i <= 200; ++i) {
        const double x = 0.05 * static_cast<double>(i);
        // The right side is tighter, so a bounded target shift should move
        // left (positive vehicle-frame y) without exceeding 10 cm.
        path.push_back(point(x, x, 0.0, 0.0, 0.0, 3.0, 1.0, 0.2));
    }

    PurePursuit controller(config);
    controller.setTrajectory(path);
    VehicleState state;
    state.pose = Pose2D(1.0, 0.0, 0.0);
    state.velocity = 3.0;
    const auto output = controller.compute(state);
    assert_finite_command(output);
    require(output.steering_angle > 0.0,
            "narrow-side corridor bias selected the wrong direction");
    require(output.steering_angle < 0.5236,
            "corridor bias bypassed the steering bound");
}

struct OffTrackPlant {
    Pose2D pose;
    double speed{0.0};
    double steering{0.0};

    void step(const PurePursuitOutput& command) {
        const double steering_step = kMaxSteeringRate * kDt;
        const double steering_error = command.steering_angle - steering;
        steering += std::clamp(steering_error, -steering_step, steering_step);
        steering = std::clamp(steering, -kMaxSteering, kMaxSteering);

        // This is a deliberately simple off-track regression plant. It is
        // not a simulator calibration model: it only exercises the command
        // contract with a bounded 10 Hz actuator response. Real feed-forward
        // values must still come from measured AutoDRIVE calibration data.
        const double desired_accel = (command.target_speed - speed) / 0.35;
        const double accel = std::clamp(desired_accel, -8.0, 3.0);
        speed = std::max(0.0, speed + accel * kDt);

        const double yaw_rate = speed / kWheelbase * std::tan(steering);
        pose.theta = std::atan2(
            std::sin(pose.theta + yaw_rate * kDt),
            std::cos(pose.theta + yaw_rate * kDt));
        pose.x += speed * std::cos(pose.theta) * kDt;
        pose.y += speed * std::sin(pose.theta) * kDt;
    }
};

void test_offtrack_10_hz_bicycle_reaches_high_speed() {
    PurePursuitConfig config;
    config.wall_bias_gain = 0.0;
    config.max_lateral_accel = 6.5;

    std::vector<TrajectoryPoint> path;
    for (int i = 0; i <= 2400; ++i) {
        const double x = 0.05 * static_cast<double>(i);
        path.push_back(point(x, x, 0.0, 0.0, 0.0, 5.5));
    }

    PurePursuit controller(config);
    controller.setTrajectory(path);
    OffTrackPlant plant;
    plant.pose = Pose2D(0.5, 0.0, 0.0);

    double max_speed = 0.0;
    double max_abs_cte = 0.0;
    for (int step = 0; step < 100; ++step) {
        VehicleState state;
        state.pose = plant.pose;
        state.velocity = plant.speed;
        state.steering_angle = plant.steering;
        const auto output = controller.compute(state);
        assert_finite_command(output);
        require(output.target_speed > 5.0,
                "off-track high-speed straight target fell below 5 m/s");
        max_abs_cte = std::max(max_abs_cte, std::abs(output.cross_track_error));
        plant.step(output);
        max_speed = std::max(max_speed, plant.speed);
    }

    require(max_speed > 5.0,
            "off-track 10 Hz plant did not reach 5 m/s");
    require(max_abs_cte < 0.05,
            "off-track straight accumulated cross-track error");
}

void test_offtrack_10_hz_turn_with_steering_lag() {
    PurePursuitConfig config;
    config.wall_bias_gain = 0.0;
    config.max_lateral_accel = 6.5;

    constexpr double radius = 12.0;
    constexpr int samples = 480;
    std::vector<TrajectoryPoint> path;
    for (int i = 0; i < samples; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) / samples;
        path.push_back(point(
            radius * angle,
            radius * std::cos(angle),
            radius * std::sin(angle),
            angle + 0.5 * kPi,
            1.0 / radius,
            5.5));
    }

    PurePursuit controller(config);
    controller.setTrajectory(path);
    OffTrackPlant plant;
    plant.pose = Pose2D(radius, 0.0, 0.5 * kPi);

    double max_speed = 0.0;
    double max_radial_error = 0.0;
    for (int step = 0; step < 220; ++step) {
        VehicleState state;
        state.pose = plant.pose;
        state.velocity = plant.speed;
        state.steering_angle = plant.steering;
        const auto output = controller.compute(state);
        assert_finite_command(output);
        plant.step(output);

        if (step > 20) {
            const double radial_error =
                std::abs(std::hypot(plant.pose.x, plant.pose.y) - radius);
            max_radial_error = std::max(max_radial_error, radial_error);
        }
        max_speed = std::max(max_speed, plant.speed);
    }

    require(max_speed > 5.0,
            "off-track turn never reached the requested high-speed regime");
    require(max_radial_error < 0.75,
            "10 Hz steering lag made the off-track turn diverge");
}

void test_invalid_and_empty_paths_fail_safe() {
    PurePursuitConfig config;
    PurePursuit controller(config);
    VehicleState state;
    const auto empty_output = controller.compute(state);
    require(!empty_output.valid, "empty trajectory produced a valid command");

    std::vector<TrajectoryPoint> short_path;
    short_path.push_back(point(0.0, 0.0, 0.0, 0.0, 0.0, 1.0));
    short_path.push_back(point(1.0, 1.0, 0.0, 0.0, 0.0, 1.0));
    controller.setTrajectory(short_path);
    const auto short_output = controller.compute(state);
    require(short_output.valid,
            "a two-point open path should remain usable in memory");
    assert_finite_command(short_output);
}

}  // namespace

int main() {
    try {
        test_five_point_five_metre_per_second_straight();
        test_ten_hz_straight_closed_loop();
        test_positive_curvature_turn();
        test_preview_brakes_before_slow_section();
        test_narrow_corridor_bias_is_bounded();
        test_offtrack_10_hz_bicycle_reaches_high_speed();
        test_offtrack_10_hz_turn_with_steering_lag();
        test_invalid_and_empty_paths_fail_safe();
        std::cout << "pure_pursuit_offtrack_suite: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pure_pursuit_offtrack_suite: FAIL: " << error.what() << '\n';
        return 1;
    }
}

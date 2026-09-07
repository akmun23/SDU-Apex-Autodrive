/**
 * Offline track-level Pure Pursuit parameter sweep.
 *
 * This is a screening tool, not a replacement for simulator acceptance. It
 * runs the production PurePursuit implementation against the exported
 * raceline and a bounded kinematic/servo plant. The plant deliberately uses
 * the documented simulator geometry and steering-rate limit, so controller
 * candidates can be rejected before another live run.
 */

#include "algorithms/pure_pursuit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using f1tenth_control::Point2D;
using f1tenth_control::Pose2D;
using f1tenth_control::PurePursuit;
using f1tenth_control::PurePursuitConfig;
using f1tenth_control::PurePursuitOutput;
using f1tenth_control::TrajectoryPoint;
using f1tenth_control::VehicleState;

constexpr double kDefaultDt = 0.05;              // measured compete cadence
constexpr double kDefaultWheelbase = 0.324;
constexpr double kDefaultMaxSteering = 0.5236;
constexpr double kDefaultSteeringRate = 3.2;
constexpr double kDefaultServoTau = 0.117;
constexpr double kDefaultPassiveDecel = 1.50;
constexpr double kDefaultMaxAccel = 3.0;
constexpr double kDefaultSpeedCap = 12.0;
constexpr double kVehicleHalfWidth = 0.1365;
constexpr double kWallMargin = 0.03;

struct Candidate {
    // Pure Pursuit parameters only.  Vehicle/actuator values live in Plant
    // below and are deliberately not part of the search space.
    double min_lookahead{0.35};
    double max_lookahead{0.95};
    double lookahead_gain{0.05};
    double cte_lookahead_weight{1.0};
    double cte_lookahead_gain{0.041540516};
    double curvature_lookahead_gain{1.9003721};
    double curvature_speed_factor{0.1015252};
    double curvature_speed_floor_ratio{0.52401066};
    double cte_speed_factor{1.50};
    double cte_speed_floor_ratio{0.55};
    double curvature_preview_factor{1.6245233};
    double curvature_feedforward_gain{0.50};
    double heading_error_gain{0.30};
    double speed_preview_distance{4.0};
    double speed_profile_braking_decel{1.50};
    double corridor_half_width_ref{0.35};
    double corridor_speed_floor_ratio{0.25};
    double corridor_lookahead_factor{2.0};
    double wall_bias_gain{0.25};
    double wall_bias_max_m{0.10};
};

struct Result {
    Candidate candidate{};
    bool completed_lap{false};
    bool wall_violation{false};
    bool offtrack{false};
    int steps{0};
    double progress_m{0.0};
    double max_abs_cte{0.0};
    double mean_abs_cte{0.0};
    double mean_speed{0.0};
    double max_speed{0.0};
    double elapsed_s{0.0};
    double score{std::numeric_limits<double>::infinity()};
};

struct Plant {
    Pose2D pose;
    double speed{0.0};
    double steering{0.0};
    double wheelbase{kDefaultWheelbase};
    double max_steering{kDefaultMaxSteering};
    double steering_rate{kDefaultSteeringRate};
    double servo_tau{kDefaultServoTau};
    double passive_decel{kDefaultPassiveDecel};
    double max_accel{kDefaultMaxAccel};
    double dt{kDefaultDt};

    void step(double target_speed, double target_steering) {
        target_speed = std::max(0.0, target_speed);
        target_steering = std::clamp(
            target_steering, -max_steering, max_steering);

        // First-order servo response bounded by the documented centre-steering
        // rate. This separates steering lag from the controller's geometry.
        const double desired_rate = (target_steering - steering) /
                                     std::max(1.0e-3, servo_tau);
        const double bounded_rate = std::clamp(
            desired_rate, -steering_rate, steering_rate);
        steering = std::clamp(
            steering + bounded_rate * dt, -max_steering, max_steering);

        if (target_speed >= speed) {
            speed = std::min(target_speed, speed + max_accel * dt);
        } else {
            // The available command path has neutral/coast semantics rather
            // than an active brake channel. Keep that limitation explicit.
            speed = std::max(target_speed, speed - passive_decel * dt);
        }

        const double yaw_rate = speed * std::tan(steering) / wheelbase;
        const double mid_heading = pose.theta + 0.5 * yaw_rate * dt;
        pose.x += speed * std::cos(mid_heading) * dt;
        pose.y += speed * std::sin(mid_heading) * dt;
        pose.theta = std::atan2(
            std::sin(pose.theta + yaw_rate * dt),
            std::cos(pose.theta + yaw_rate * dt));
    }
};

void usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " --trajectory PATH [--start-index N] [--speed-cap MPS]"
              << " [--laps N] [--dt SEC] [--baseline] [--csv PATH]\n";
}

bool parse_double(const std::string& value, double& output) {
    char* end = nullptr;
    output = std::strtod(value.c_str(), &end);
    return end != value.c_str() && *end == '\0' && std::isfinite(output);
}

bool parse_int(const std::string& value, int& output) {
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        return false;
    }
    output = static_cast<int>(parsed);
    return true;
}

Result run_candidate(
    const std::string& trajectory_path,
    const Candidate& candidate,
    int start_index,
    double speed_cap,
    int laps,
    double dt) {
    Result result;
    result.candidate = candidate;

    PurePursuitConfig config;
    config.min_lookahead = candidate.min_lookahead;
    config.max_lookahead = candidate.max_lookahead;
    config.lookahead_gain = candidate.lookahead_gain;
    config.cte_lookahead_weight = candidate.cte_lookahead_weight;
    config.cte_lookahead_gain = candidate.cte_lookahead_gain;
    config.curvature_lookahead_gain = candidate.curvature_lookahead_gain;
    config.curvature_speed_factor = candidate.curvature_speed_factor;
    config.curvature_speed_floor_ratio = candidate.curvature_speed_floor_ratio;
    config.cte_speed_factor = candidate.cte_speed_factor;
    config.cte_speed_floor_ratio = candidate.cte_speed_floor_ratio;
    config.curvature_preview_factor = candidate.curvature_preview_factor;
    config.curvature_feedforward_gain = candidate.curvature_feedforward_gain;
    config.heading_error_gain = candidate.heading_error_gain;
    config.speed_preview_distance = candidate.speed_preview_distance;
    config.speed_profile_braking_decel = candidate.speed_profile_braking_decel;
    config.corridor_half_width_ref = candidate.corridor_half_width_ref;
    config.corridor_speed_floor_ratio = candidate.corridor_speed_floor_ratio;
    config.corridor_lookahead_factor = candidate.corridor_lookahead_factor;
    config.wall_bias_gain = candidate.wall_bias_gain;
    config.wall_bias_max_m = candidate.wall_bias_max_m;
    config.min_regulated_speed = 0.0;
    config.max_command_speed = speed_cap;
    config.offtrack_stop_error_m = 0.75;
    config.max_lateral_accel = 6.5;

    PurePursuit controller(config);
    if (!controller.loadTrajectory(trajectory_path)) {
        result.offtrack = true;
        return result;
    }
    const auto& trajectory = controller.getTrajectory();
    if (trajectory.size() < 3) {
        result.offtrack = true;
        return result;
    }

    const int n = static_cast<int>(trajectory.size());
    start_index = ((start_index % n) + n) % n;
    const double track_length = trajectory.back().arc_length;
    const double start_s = trajectory[static_cast<size_t>(start_index)].arc_length;

    Plant plant;
    plant.dt = dt;
    plant.pose = Pose2D(
        trajectory[static_cast<size_t>(start_index)].x,
        trajectory[static_cast<size_t>(start_index)].y,
        trajectory[static_cast<size_t>(start_index)].heading);

    // Warm the controller at the actual spawn progress. The first call uses
    // its full-search recovery path; subsequent calls use progress tracking.
    double previous_s = start_s;
    double accumulated_progress = 0.0;
    double cte_sum = 0.0;
    double speed_sum = 0.0;
    const double required_progress = std::max(1.0, track_length * laps);
    // The command path can coast/brake and the raceline may be much faster
    // than the fixed plant envelope.  Budget against a conservative quarter
    // of the requested cap, with a minimum absolute average speed, so a
    // candidate is only rejected for its controller behavior—not because the
    // harness stopped before it could finish the lap.
    const double conservative_average_speed = std::max(0.5, speed_cap * 0.25);
    const int max_steps = static_cast<int>(std::ceil(
        required_progress / conservative_average_speed / dt * 2.0));

    for (int step = 0; step < max_steps; ++step) {
        VehicleState state;
        state.pose = plant.pose;
        state.velocity = plant.speed;
        state.angular_velocity = plant.speed *
            std::tan(plant.steering) / plant.wheelbase;
        state.steering_angle = plant.steering;

        const PurePursuitOutput output = controller.compute(state);
        if (!output.valid || !std::isfinite(output.cross_track_error)) {
            result.offtrack = true;
            break;
        }

        const auto& closest = trajectory[output.closest_idx];
        double delta_s = closest.arc_length - previous_s;
        if (delta_s < -0.5 * track_length) {
            delta_s += track_length;
        }
        if (delta_s >= 0.0 && delta_s < 0.25 * track_length) {
            accumulated_progress += delta_s;
            previous_s = closest.arc_length;
        }

        const double abs_cte = std::abs(output.cross_track_error);
        result.max_abs_cte = std::max(result.max_abs_cte, abs_cte);
        cte_sum += abs_cte;
        result.mean_abs_cte = cte_sum / static_cast<double>(step + 1);
        speed_sum += plant.speed;
        result.mean_speed = speed_sum / static_cast<double>(step + 1);
        result.max_speed = std::max(result.max_speed, plant.speed);
        result.steps = step + 1;
        result.elapsed_s = static_cast<double>(step + 1) * dt;
        result.progress_m = accumulated_progress;

        const double left_clearance = closest.left_bound - output.cross_track_error;
        const double right_clearance = closest.right_bound + output.cross_track_error;
        if (left_clearance < kVehicleHalfWidth + kWallMargin ||
            right_clearance < kVehicleHalfWidth + kWallMargin) {
            result.wall_violation = true;
        }
        if (abs_cte > config.offtrack_stop_error_m) {
            result.offtrack = true;
            break;
        }

        const double target_speed = std::clamp(
            output.target_speed, 0.0, speed_cap);
        plant.step(target_speed, output.steering_angle);

        if (accumulated_progress >= required_progress) {
            result.completed_lap = true;
            break;
        }
    }

    // Successful candidates are ranked by tracking quality. Failed candidates
    // remain visible in the report but cannot win the selection.
    // Feasible candidates are ranked by tracking first, then by the ability to
    // retain speed.  This is a PP tuning score; it does not identify or alter
    // the vehicle plant.
    result.score = 10.0 * result.mean_abs_cte + result.max_abs_cte +
        0.01 * result.elapsed_s - 0.02 * result.mean_speed;
    if (result.wall_violation) {
        result.score += 25.0;
    }
    if (result.offtrack || !result.completed_lap) {
        result.score += 1000.0 +
            std::max(0.0, required_progress - result.progress_m);
    }
    return result;
}

bool better(const Result& lhs, const Result& rhs) {
    const bool lhs_pass = lhs.completed_lap && !lhs.wall_violation && !lhs.offtrack;
    const bool rhs_pass = rhs.completed_lap && !rhs.wall_violation && !rhs.offtrack;
    if (lhs_pass != rhs_pass) {
        return lhs_pass;
    }
    return lhs.score < rhs.score;
}

}  // namespace

int main(int argc, char** argv) {
    std::string trajectory_path;
    std::string csv_path;
    int start_index = 0;
    int laps = 1;
    double speed_cap = kDefaultSpeedCap;
    double dt = kDefaultDt;
    bool sweep = true;

    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--trajectory" && i + 1 < argc) {
            trajectory_path = argv[++i];
        } else if (argument == "--start-index" && i + 1 < argc) {
            if (!parse_int(argv[++i], start_index)) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--speed-cap" && i + 1 < argc) {
            if (!parse_double(argv[++i], speed_cap) || speed_cap <= 0.0) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--laps" && i + 1 < argc) {
            if (!parse_int(argv[++i], laps) || laps < 1 || laps > 5) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--dt" && i + 1 < argc) {
            if (!parse_double(argv[++i], dt) || dt <= 0.0 || dt > 0.2) {
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--baseline") {
            sweep = false;
        } else if (argument == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (trajectory_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    std::vector<Result> results;

    const Candidate baseline{};
    results.push_back(run_candidate(
        trajectory_path, baseline, start_index, speed_cap, laps, dt));

    if (sweep) {
        // First vary each PP parameter around the production baseline.  This
        // makes the CSV useful for diagnosing which mechanism is helping,
        // rather than only returning an opaque winner.
        const auto one_at_a_time = [&](auto setter, const std::vector<double>& values) {
            for (const double value : values) {
                Candidate candidate = baseline;
                setter(candidate, value);
                if (candidate.max_lookahead + 1.0e-9 >= candidate.min_lookahead) {
                    results.push_back(run_candidate(
                        trajectory_path, candidate, start_index, speed_cap, laps, dt));
                }
            }
        };

        one_at_a_time([](Candidate& c, double v) { c.min_lookahead = v; }, {0.35, 0.50, 0.65, 0.80});
        one_at_a_time([](Candidate& c, double v) { c.max_lookahead = v; }, {0.85, 1.15, 1.45, 1.80});
        one_at_a_time([](Candidate& c, double v) { c.lookahead_gain = v; }, {0.03, 0.07, 0.14, 0.22});
        one_at_a_time([](Candidate& c, double v) { c.cte_lookahead_weight = v; }, {0.0, 0.5, 1.0, 1.5});
        one_at_a_time([](Candidate& c, double v) { c.cte_lookahead_gain = v; }, {0.0, 0.04, 0.08, 0.15});
        one_at_a_time([](Candidate& c, double v) { c.curvature_lookahead_gain = v; }, {0.8, 1.3, 1.9, 2.8});
        one_at_a_time([](Candidate& c, double v) { c.curvature_speed_factor = v; }, {0.05, 0.10, 0.20, 0.35});
        one_at_a_time([](Candidate& c, double v) { c.curvature_speed_floor_ratio = v; }, {0.35, 0.52, 0.70, 0.85});
        one_at_a_time([](Candidate& c, double v) { c.cte_speed_factor = v; }, {0.5, 1.5, 2.5, 4.0});
        one_at_a_time([](Candidate& c, double v) { c.cte_speed_floor_ratio = v; }, {0.35, 0.55, 0.75, 0.90});
        one_at_a_time([](Candidate& c, double v) { c.curvature_preview_factor = v; }, {1.0, 1.6, 2.4, 3.2});
        one_at_a_time([](Candidate& c, double v) { c.curvature_feedforward_gain = v; }, {0.0, 0.25, 0.50, 0.75, 1.0});
        one_at_a_time([](Candidate& c, double v) { c.heading_error_gain = v; }, {0.0, 0.15, 0.30, 0.45, 0.60});
        one_at_a_time([](Candidate& c, double v) { c.speed_preview_distance = v; }, {2.0, 4.0, 6.0, 8.0});
        one_at_a_time([](Candidate& c, double v) { c.speed_profile_braking_decel = v; }, {0.75, 1.50, 2.50, 4.0});
        one_at_a_time([](Candidate& c, double v) { c.corridor_half_width_ref = v; }, {0.25, 0.35, 0.50, 0.75});
        one_at_a_time([](Candidate& c, double v) { c.corridor_speed_floor_ratio = v; }, {0.15, 0.25, 0.40, 0.60});
        one_at_a_time([](Candidate& c, double v) { c.corridor_lookahead_factor = v; }, {0.0, 1.0, 2.0, 3.0});
        one_at_a_time([](Candidate& c, double v) { c.wall_bias_gain = v; }, {0.0, 0.25, 0.50, 1.0});
        one_at_a_time([](Candidate& c, double v) { c.wall_bias_max_m = v; }, {0.0, 0.05, 0.10, 0.15});

        // Then test the coupled effects that are easy to miss with isolated
        // changes: adaptive lookahead, curvature turn-in, and speed preview.
        for (const double min_lh : {0.45, 0.65, 0.85}) {
            for (const double gain : {0.05, 0.14, 0.22}) {
                for (const double ff : {0.0, 0.25, 0.50}) {
                    Candidate candidate = baseline;
                    candidate.min_lookahead = min_lh;
                    candidate.max_lookahead = min_lh + 0.50;
                    candidate.lookahead_gain = gain;
                    candidate.curvature_feedforward_gain = ff;
                    results.push_back(run_candidate(
                        trajectory_path, candidate, start_index, speed_cap, laps, dt));
                }
            }
        }

        for (const double preview_factor : {1.0, 1.6, 2.4}) {
            for (const double curv_factor : {0.05, 0.10, 0.20}) {
                for (const double preview_distance : {2.0, 4.0, 6.0}) {
                    Candidate candidate = baseline;
                    candidate.curvature_preview_factor = preview_factor;
                    candidate.curvature_speed_factor = curv_factor;
                    candidate.speed_preview_distance = preview_distance;
                    results.push_back(run_candidate(
                        trajectory_path, candidate, start_index, speed_cap, laps, dt));
                }
            }
        }
    }

    std::sort(results.begin(), results.end(), better);
    std::ofstream csv_file;
    std::ostream* csv = &std::cout;
    if (!csv_path.empty()) {
        csv_file.open(csv_path);
        if (!csv_file.is_open()) {
            std::cerr << "ERROR: cannot open CSV output: " << csv_path << '\n';
            return 2;
        }
        csv = &csv_file;
    }
    *csv << "candidate,min_lookahead,max_lookahead,lookahead_gain,"
            "cte_lookahead_weight,cte_lookahead_gain,curvature_lookahead_gain,"
            "curvature_speed_factor,curvature_speed_floor_ratio,cte_speed_factor,"
            "cte_speed_floor_ratio,curvature_preview_factor,curvature_feedforward_gain,"
            "heading_error_gain,"
            "speed_preview_distance,speed_profile_braking_decel,corridor_half_width_ref,"
            "corridor_speed_floor_ratio,corridor_lookahead_factor,wall_bias_gain,wall_bias_max_m,"
            "completed_lap,wall_violation,offtrack,steps,progress_m,max_abs_cte,mean_abs_cte,"
            "mean_speed,max_speed,elapsed_s,score\n";
    for (const auto& result : results) {
        *csv << std::fixed << std::setprecision(6)
                  << "candidate," << result.candidate.min_lookahead << ','
                  << result.candidate.max_lookahead << ','
                  << result.candidate.lookahead_gain << ','
                  << result.candidate.cte_lookahead_weight << ','
                  << result.candidate.cte_lookahead_gain << ','
                  << result.candidate.curvature_lookahead_gain << ','
                  << result.candidate.curvature_speed_factor << ','
                  << result.candidate.curvature_speed_floor_ratio << ','
                  << result.candidate.cte_speed_factor << ','
                  << result.candidate.cte_speed_floor_ratio << ','
                  << result.candidate.curvature_preview_factor << ','
                  << result.candidate.curvature_feedforward_gain << ','
                  << result.candidate.heading_error_gain << ','
                  << result.candidate.speed_preview_distance << ','
                  << result.candidate.speed_profile_braking_decel << ','
                  << result.candidate.corridor_half_width_ref << ','
                  << result.candidate.corridor_speed_floor_ratio << ','
                  << result.candidate.corridor_lookahead_factor << ','
                  << result.candidate.wall_bias_gain << ','
                  << result.candidate.wall_bias_max_m << ','
                  << (result.completed_lap ? 1 : 0) << ','
                  << (result.wall_violation ? 1 : 0) << ','
                  << (result.offtrack ? 1 : 0) << ','
                  << result.steps << ',' << result.progress_m << ','
                  << result.max_abs_cte << ',' << result.mean_abs_cte << ','
                  << result.mean_speed << ',' << result.max_speed << ','
                  << result.elapsed_s << ',' << result.score << '\n';
    }

    if (results.empty()) {
        return 1;
    }
    const auto& best = results.front();
    std::cerr << "BEST min_lookahead=" << best.candidate.min_lookahead
              << " max_lookahead=" << best.candidate.max_lookahead
              << " lookahead_gain=" << best.candidate.lookahead_gain
              << " cte_lookahead_gain=" << best.candidate.cte_lookahead_gain
              << " curvature_preview_factor=" << best.candidate.curvature_preview_factor
              << " curvature_feedforward_gain=" << best.candidate.curvature_feedforward_gain
              << " heading_error_gain=" << best.candidate.heading_error_gain
              << " cte_speed_factor=" << best.candidate.cte_speed_factor
              << " curvature_speed_factor=" << best.candidate.curvature_speed_factor
              << " completed_lap=" << (best.completed_lap ? "true" : "false")
              << " wall_violation=" << (best.wall_violation ? "true" : "false")
              << " max_abs_cte=" << best.max_abs_cte
              << " mean_abs_cte=" << best.mean_abs_cte
              << " mean_speed=" << best.mean_speed
              << " score=" << best.score << '\n';
    return (best.completed_lap && !best.wall_violation && !best.offtrack) ? 0 : 1;
}

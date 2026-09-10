#include "algorithms/follow_the_gap.hpp"

namespace f1tenth_control {

// =====================================================================
// Construction / configuration
// =====================================================================

FollowTheGap::FollowTheGap(const FTGConfig& config)
    : config_(config), lidar_processor_(config.lidar_config), last_steering_(0.0) {}

void FollowTheGap::setConfig(const FTGConfig& config) {
    config_ = config;
    lidar_processor_.setConfig(config.lidar_config);
}

void FollowTheGap::reset() {
    last_steering_ = 0.0;
    smoothed_target_ = 0.0;
    first_compute_ = true;
    last_source_stamp_s_ = 0.0;
    has_source_stamp_ = false;
    has_selected_gap_ = false;
    previous_gap_angle_ = 0.0;
    previous_gap_score_ = 0.0;
    pending_alternative_count_ = 0;
}

// =====================================================================
// Main compute
// =====================================================================

FTGOutput FollowTheGap::compute(
    const std::vector<float>& ranges,
    double angle_min,
    double angle_max,
    double angle_increment,
    double source_stamp_s
) {
    FTGOutput output;
    output.source_stamp_s = std::isfinite(source_stamp_s) ? source_stamp_s : 0.0;

    // Steering-rate limiting is driven by the LiDAR source clock, never by
    // executor scheduling. Calls without a source stamp use a fixed test step.
    const double dt = computeDeltaTime(source_stamp_s);

    // --- Handle empty scan ---
    if (ranges.empty()) {
        output.emergency_stop = true;
        output.no_path = true;
        output.command = DriveCommand(0.0, 0.0);
        return output;
    }

    double raw_closest_range = std::numeric_limits<double>::infinity();
    const bool raw_emergency = rawEmergency(ranges, raw_closest_range);
    output.closest_point_dist = raw_closest_range;

    // --- Step 1: Generic LiDAR preprocessing (median filter, range clip) ---
    ProcessedScan scan = lidar_processor_.processScan(
        ranges, angle_min, angle_max, angle_increment
    );

    // If no valid points after preprocessing, trigger emergency stop
    if (scan.filtered_ranges.empty()) {
        output.emergency_stop = true;
        output.no_path = true;
        output.command = DriveCommand(0.0, 0.0);
        return output;
    }

    // Initialise disparity markers used by the safety extension.
    scan.disparity_blocked.assign(scan.filtered_ranges.size(), false);

    // --- Step 2: Wall margin  ---
    applyWallMargin(scan);

    // --- Step 3: Closest-point detection ---
    const size_t closest_point_idx = lidar_processor_.findClosestPoint(scan);
    if (!std::isfinite(output.closest_point_dist)) {
        output.closest_point_dist = scan.filtered_ranges[closest_point_idx];
    }

    // Check raw returns before range_min/median preprocessing can hide a close
    // obstacle. This is independent from navigation clearance processing.
    if (raw_emergency) {
        output.emergency_stop = true;
        output.command = DriveCommand(0.0, 0.0);
        return output;
    }

    // --- Step 4: Disparity extension ---
    applyDisparityExtension(scan);

    // --- Step 5: Compute effective clearance per beam ---
    std::vector<double> eff_clearance = computeEffectiveClearance(scan);

    // --- Step 6: Find and select exactly one connected control gap ---
    output.drivable_gaps = findDrivableGaps(scan, eff_clearance);
    if (output.drivable_gaps.empty()) {
        has_selected_gap_ = false;
        pending_alternative_count_ = 0;
        output.no_path = true;
        output.command = DriveCommand(0.0, 0.0);
        return output;
    }

    output.selected_drivable_gap = selectDrivableGap(output.drivable_gaps);
    output.has_selected_drivable_gap = true;
    const TargetResult target = computeTargetAngle(output.selected_drivable_gap);
    if (!target.valid) {
        output.no_path = true;
        output.command = DriveCommand(0.0, 0.0);
        return output;
    }
    output.raw_target_angle = target.angle;

    // --- Step 7: EMA smoothing on the target angle ---
    if (first_compute_) {
        smoothed_target_ = target.angle;
        first_compute_ = false;
    } else {
        const double alpha = std::clamp(config_.target_ema_alpha, 0.0, 1.0);
        smoothed_target_ = alpha * target.angle + (1.0 - alpha) * smoothed_target_;
    }
    output.smoothed_target_angle = smoothed_target_;

    // Raw demand is retained separately because speed must react before the
    // physical steering-rate limit catches up.
    const double desired_steering = std::clamp(
        config_.steering_gain * smoothed_target_,
        -config_.max_steering,
        config_.max_steering
    );

    // Validate the requested branch against the vehicle footprint, not merely
    // against individual LiDAR rays. A target can be ray-clear while the
    // swept car body still clips the inside of a hairpin.
    const TrajectoryResult trajectory = selectTrajectory(
        ranges,
        angle_min,
        angle_increment,
        output.selected_drivable_gap,
        desired_steering);
    output.trajectory_free_distance = trajectory.free_distance;
    output.trajectory_min_clearance = trajectory.min_clearance;
    output.trajectory_collision_free = trajectory.collision_free;
    if (!trajectory.valid) {
        output.no_path = true;
        output.command = DriveCommand(0.0, 0.0);
        last_steering_ = 0.0;
        return output;
    }

    const double raw_steering = trajectory.steering;
    output.raw_steering = raw_steering;

    // --- Step 8: Time-based steering rate limiting ---
    double steering = rateLimitSteering(raw_steering, last_steering_, dt);
    last_steering_ = steering;
    output.rate_limited_steering = steering;

    // --- Step 9: Speed from forward clearance + steering ---
    // Forward clearance: average effective clearance in the central ±10 deg cone
    double fwd_clearance = 0.0;
    int fwd_count = 0;
    const double fwd_cone = 0.175;  // ~10 degrees

    // Loop through all beams and average effective clearance 
    // for valid beams within the forward cone
    for (size_t i = 0; i < scan.filtered_ranges.size(); ++i) {
        if (std::abs(scan.angles[i]) <= fwd_cone && scan.valid[i]) {
            fwd_clearance += eff_clearance[i];
            ++fwd_count;
        }
    }
    if (fwd_count > 0) {
        fwd_clearance /= static_cast<double>(fwd_count);
    }
    // The rollout is the authoritative forward safety horizon. This also
    // handles scans whose usable navigation sector has no central beam.
    if (trajectory.free_distance > 0.0) {
        fwd_clearance = fwd_count > 0
            ? std::min(fwd_clearance, trajectory.free_distance)
            : trajectory.free_distance;
    }
    output.forward_clearance = fwd_clearance;

    // Use demanded steering for anticipatory slowdown while the actuator is
    // still ramping toward the requested turn.
    const double speed_steering = std::max(std::abs(raw_steering), std::abs(steering));
    double speed = calculateSpeed(fwd_clearance, speed_steering);
    if (!trajectory.collision_free) {
        // A partial rollout is usable only as a cautious escape/cornering
        // command. Never let the normal mapping speed exceed the distance
        // that the swept footprint has actually verified.
        speed = std::min(
            speed,
            std::max(config_.min_speed, 0.5 * trajectory.free_distance));
    }
    output.command = DriveCommand(speed, steering);

    return output;
}

FollowTheGap::TrajectoryResult FollowTheGap::selectTrajectory(
    const std::vector<float>& ranges,
    double angle_min,
    double angle_increment,
    const DrivableGap& selected_gap,
    double desired_steering
) const {
    TrajectoryResult best;
    TrajectoryResult best_observed;
    best.free_distance = 0.0;
    best.min_clearance = 0.0;

    if (ranges.empty()
        || !std::isfinite(angle_min)
        || !std::isfinite(angle_increment)
        || std::abs(angle_increment) < 1e-9
        || config_.wheelbase <= 0.0
        || config_.rollout_distance <= 0.0
        || config_.rollout_step <= 0.0
        || config_.trajectory_candidate_count < 1) {
        return best;
    }

    struct Point {
        double x;
        double y;
    };
    std::vector<Point> obstacles;
    obstacles.reserve(ranges.size());
    const double min_range = std::max(0.0, config_.lidar_config.range_min);
    const double max_range = std::max(min_range, config_.lidar_config.range_max);
    for (size_t i = 0; i < ranges.size(); ++i) {
        const double range = static_cast<double>(ranges[i]);
        if (!std::isfinite(range) || range < min_range || range > max_range) {
            continue;
        }
        const double angle = angle_min + static_cast<double>(i) * angle_increment;
        if (!std::isfinite(angle)) continue;
        obstacles.push_back({
            config_.lidar_to_rear_axle + range * std::cos(angle),
            range * std::sin(angle),
        });
    }

    const int candidate_count = std::max(1, config_.trajectory_candidate_count);
    const double half_width = 0.5 * config_.car_width
        + std::max(0.0, config_.footprint_margin);
    const double x_min = -std::max(0.0, config_.rear_overhang)
        - std::max(0.0, config_.footprint_margin);
    const double x_max = config_.car_length
        - std::max(0.0, config_.rear_overhang)
        + std::max(0.0, config_.footprint_margin);
    const double target = std::clamp(
        desired_steering, -config_.max_steering, config_.max_steering);
    const TargetResult branch_target = computeTargetAngle(selected_gap);
    const double gap_start = std::min(selected_gap.start_angle, selected_gap.end_angle);
    const double gap_end = std::max(selected_gap.start_angle, selected_gap.end_angle);
    const double branch_margin = 0.05;
    const double validation_distance = std::min(0.75, config_.rollout_distance);
    const double maximum_reachable_heading = std::abs(
        std::tan(config_.max_steering) / config_.wheelbase)
        * validation_distance;
    const bool gap_reachable_at_validation = gap_start
        <= maximum_reachable_heading + branch_margin
        && gap_end >= -maximum_reachable_heading - branch_margin;
    const bool material_branch = branch_target.valid
        && std::abs(branch_target.angle) > 0.35;

    auto better = [&](const TrajectoryResult& candidate,
                      const TrajectoryResult& incumbent) {
        if (!incumbent.valid) return true;
        const double candidate_progress = candidate.free_distance
            / std::max(config_.rollout_distance, 1e-9);
        const double incumbent_progress = incumbent.free_distance
            / std::max(config_.rollout_distance, 1e-9);
        if (std::abs(candidate_progress - incumbent_progress) > 1e-9) {
            return candidate_progress > incumbent_progress;
        }
        const double candidate_target_error = std::abs(candidate.steering - target);
        const double incumbent_target_error = std::abs(incumbent.steering - target);
        if (std::abs(candidate_target_error - incumbent_target_error) > 1e-9) {
            return candidate_target_error < incumbent_target_error;
        }
        return candidate.min_clearance > incumbent.min_clearance;
    };

    for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        const double fraction = candidate_count == 1
            ? 0.5
            : static_cast<double>(candidate_index)
                / static_cast<double>(candidate_count - 1);
        const double steering = -config_.max_steering
            + 2.0 * config_.max_steering * fraction;
        const double curvature = std::tan(steering) / config_.wheelbase;

        double free_distance = config_.rollout_distance;
        double min_clearance = std::numeric_limits<double>::infinity();
        bool collision = false;
        const int steps = std::max(
            1, static_cast<int>(std::ceil(
                config_.rollout_distance / config_.rollout_step)));
        for (int step = 0; step <= steps; ++step) {
            const double distance = std::min(
                config_.rollout_distance,
                static_cast<double>(step) * config_.rollout_step);
            double pose_x = distance;
            double pose_y = 0.0;
            double pose_yaw = 0.0;
            if (std::abs(curvature) > 1e-9) {
                pose_x = std::sin(curvature * distance) / curvature;
                pose_y = (1.0 - std::cos(curvature * distance)) / curvature;
                pose_yaw = curvature * distance;
            }
            const double cos_yaw = std::cos(pose_yaw);
            const double sin_yaw = std::sin(pose_yaw);

            for (const auto& point : obstacles) {
                const double dx = point.x - pose_x;
                const double dy = point.y - pose_y;
                const double local_x = cos_yaw * dx + sin_yaw * dy;
                const double local_y = -sin_yaw * dx + cos_yaw * dy;
                const double outside_x = std::max({x_min - local_x, 0.0, local_x - x_max});
                const double outside_y = std::max(std::abs(local_y) - half_width, 0.0);
                const double clearance = std::hypot(outside_x, outside_y);
                min_clearance = std::min(min_clearance, clearance);
                if (local_x >= x_min && local_x <= x_max
                    && std::abs(local_y) <= half_width) {
                    collision = true;
                    free_distance = std::min(free_distance, distance);
                    break;
                }
            }
            if (collision) break;
        }

        // The rollout refines the gap selected above; it must not replace it
        // with a different, deeper branch. At an early rollout point require
        // the candidate to point into the selected angular interval. For a
        // clearly lateral branch also reject the opposite steering direction.
        const double trajectory_heading = curvature * validation_distance;
        const bool enters_selected_gap = trajectory_heading >= gap_start - branch_margin
            && trajectory_heading <= gap_end + branch_margin;
        const bool branch_compatible = !gap_reachable_at_validation
            || enters_selected_gap;
        const bool follows_material_branch = !material_branch
            || steering * branch_target.angle >= -1e-9;
        if (!branch_compatible || !follows_material_branch) {
            continue;
        }

        if (!std::isfinite(min_clearance)) {
            min_clearance = config_.rollout_distance;
        }
        TrajectoryResult candidate;
        candidate.valid = free_distance
            >= std::max(0.0, config_.trajectory_min_free_distance);
        candidate.collision_free = !collision;
        candidate.steering = steering;
        candidate.free_distance = free_distance;
        candidate.min_clearance = min_clearance;
        if (better(candidate, best_observed)) best_observed = candidate;
        if (candidate.valid && better(candidate, best)) best = candidate;
    }

    if (!best.valid) return best_observed;
    return best;
}

double FollowTheGap::computeDeltaTime(double source_stamp_s) {
    constexpr double DEFAULT_DT = 0.025;
    if (!std::isfinite(source_stamp_s)) {
        return DEFAULT_DT;
    }

    double dt = DEFAULT_DT;
    if (has_source_stamp_) {
        const double source_dt = source_stamp_s - last_source_stamp_s_;
        if (source_dt > 0.0 && std::isfinite(source_dt)) {
            dt = std::clamp(source_dt, 0.001, 0.5);
        }
    }
    last_source_stamp_s_ = source_stamp_s;
    has_source_stamp_ = true;
    return dt;
}

bool FollowTheGap::rawEmergency(
    const std::vector<float>& ranges,
    double& closest_range
) const {
    closest_range = std::numeric_limits<double>::infinity();
    for (const float raw_range : ranges) {
        const double range = static_cast<double>(raw_range);
        if (!std::isfinite(range)) continue;
        closest_range = std::min(closest_range, range);
    }
    return std::isfinite(closest_range)
        && closest_range <= config_.emergency_brake_distance;
}

// =====================================================================
// LiDAR safety processing
// =====================================================================

void FollowTheGap::applyWallMargin(ProcessedScan& scan) {
    // If wall margin is zero or negative, skip this step
    if (config_.wall_margin <= 0.0) return;

    // Loop through all beams and shrink valid ranges by wall margin
    for (size_t i = 0; i < scan.filtered_ranges.size(); ++i) {
        if (scan.valid[i] && scan.filtered_ranges[i] > config_.wall_margin) {
            scan.filtered_ranges[i] -= config_.wall_margin;
        } else if (scan.valid[i]) {
            scan.filtered_ranges[i] = 0.0;
            scan.valid[i] = false;
        }
    }
}

void FollowTheGap::applyDisparityExtension(ProcessedScan& scan) {
    // If scan has fewer than 2 beams, skip disparity extension
    if (scan.filtered_ranges.size() < 2) return;

    // For convenience, create a reference to the filtered ranges vector
    std::vector<double>& ranges = scan.filtered_ranges;
    const double half_car = config_.car_width / 2.0;
    const auto& lidar_config = lidar_processor_.getConfig();

    // Cap extension to ~15 degrees worth of beams
    const int max_extension = static_cast<int>(
        std::ceil(0.26 / std::abs(scan.angle_increment))
    );

    // Loop through beams and look for large jumps in range (disparities)
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (!scan.valid[i] || !scan.valid[i - 1]) continue;

        // Check if either beam is outside the configured angular processing range
        double angle_i   = scan.angles[i];
        double angle_im1 = scan.angles[i - 1];
        if (angle_i   < lidar_config.angle_min || angle_i   > lidar_config.angle_max) continue;
        if (angle_im1 < lidar_config.angle_min || angle_im1 > lidar_config.angle_max) continue;

        // Check for large disparity
        double diff = std::abs(ranges[i] - ranges[i - 1]);
        if (diff <= config_.disparity_threshold) continue;

        // Determine which beam is closer and calculate how many neighboring beams to block
        size_t closer_idx   = (ranges[i] < ranges[i - 1]) ? i : i - 1;
        double closer_range = ranges[closer_idx];

        // Skip cascade extensions
        if (scan.disparity_blocked[closer_idx]) continue;

        // Calculate angle to extend based on geometry (car width and closer range)
        double angle_to_extend = std::atan2(half_car, closer_range);
        int indices_to_extend  = std::min(
            static_cast<int>(std::ceil(angle_to_extend / std::abs(scan.angle_increment))),
            max_extension
        );

        // Extend the closer beam's range to the farther beam and mark intermediate beams as blocked
        if (closer_idx == i) {
            // Closer on the right -> extend left
            for (int j = 0; j < indices_to_extend && static_cast<int>(i) - j >= 0; ++j) {
                size_t idx = i - j;
                if (scan.valid[idx] && ranges[idx] > closer_range) {
                    scan.disparity_blocked[idx] = true;
                    ranges[idx] = closer_range;
                }
            }
        } else {
            // Closer on the left -> extend right
            size_t base = i - 1;
            for (int j = 0; j < indices_to_extend && base + j < ranges.size(); ++j) {
                size_t idx = base + j;
                if (scan.valid[idx] && ranges[idx] > closer_range) {
                    scan.disparity_blocked[idx] = true;
                    ranges[idx] = closer_range;
                }
            }
        }
    }
}

// =====================================================================
// Weighted free-space core
// =====================================================================

std::vector<double> FollowTheGap::computeEffectiveClearance(const ProcessedScan& scan) {
    // If scan is empty, return empty effective clearance vector
    const size_t n = scan.filtered_ranges.size();
    std::vector<double> eff(n, 0.0);
    if (n == 0) return eff;

    // Precompute constants for cone calculation
    const double half_car = (config_.car_width / 2.0) * config_.clearance_cone_scale;
    const double abs_inc  = std::abs(scan.angle_increment);
    const auto& lidar_config = lidar_processor_.getConfig();

    // Loop through each beam and compute effective clearance 
    // based on minimum range in the clearance cone
    for (size_t i = 0; i < n; ++i) {
        double angle = scan.angles[i];
        // Skip beams outside the configured angular processing range or invalid beams
        if (angle < lidar_config.angle_min || angle > lidar_config.angle_max) {
            eff[i] = 0.0;
            continue;
        }
        // If beam is invalid, effective clearance is zero
        if (!scan.valid[i]) {
            eff[i] = 0.0;
            continue;
        }
        // If range is below minimum scoring range, effective clearance is zero
        double range_i = scan.filtered_ranges[i];
        if (range_i < config_.min_score_range) {
            eff[i] = 0.0;
            continue;
        }

        // Number of neighbouring beams to check (based on car width at this range)
        double cone_angle = std::atan2(half_car, range_i);
        int cone_idx = static_cast<int>(std::ceil(cone_angle / abs_inc));
        cone_idx = std::max(cone_idx, 1);  // at least 1 neighbour

        // Find minimum range in the cone ("effective" clearance)
        double min_range = range_i;
        for (int d = -cone_idx; d <= cone_idx; ++d) {
            int idx = static_cast<int>(i) + d;
            if (idx < 0 || idx >= static_cast<int>(n)) continue;
            double r = scan.filtered_ranges[idx];
            // Include invalid beams as obstacles (range 0)
            if (!scan.valid[idx]) r = 0.0;
            min_range = std::min(min_range, r);
        }

        // Effective clearance for this beam is the minimum range in its clearance cone
        eff[i] = min_range;
    }

    return eff;
}

std::vector<DrivableGap> FollowTheGap::findDrivableGaps(
    const ProcessedScan& scan,
    const std::vector<double>& eff_clearance
) const {
    std::vector<DrivableGap> gaps;
    if (scan.filtered_ranges.empty()) return gaps;

    const auto& lidar_config = lidar_processor_.getConfig();
    const size_t n = scan.filtered_ranges.size();

    auto append_gap = [&](size_t start_idx, size_t end_idx) {
        if (end_idx < start_idx) return;
        DrivableGap gap;
        gap.start_idx = start_idx;
        gap.end_idx = end_idx;
        gap.start_angle = scan.angles[start_idx];
        gap.end_angle = scan.angles[end_idx];
        gap.angular_width = std::abs(gap.end_angle - gap.start_angle);
        if (gap.angular_width < config_.min_gap_width) return;

        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        double clearance_sum = 0.0;
        double clipped_clearance_sum = 0.0;
        size_t count = 0;
        size_t deepest_idx = start_idx;
        const double decision_depth = std::max(
            config_.rollout_distance,
            config_.min_score_range);

        for (size_t i = start_idx; i <= end_idx; ++i) {
            const double clearance = eff_clearance[i];
            const double clipped_clearance = std::min(clearance, decision_depth);
            const double angle = scan.angles[i];
            const double beam_score = std::pow(
                std::max(clipped_clearance, 0.0), config_.score_power)
                * std::exp(-config_.heading_weight * std::abs(angle));
            weighted_sum += angle * beam_score;
            weight_sum += beam_score;
            clearance_sum += clearance;
            clipped_clearance_sum += clipped_clearance;
            ++count;
            if (clearance > eff_clearance[deepest_idx]
                || (clearance == eff_clearance[deepest_idx]
                    && std::abs(angle) < std::abs(scan.angles[deepest_idx]))) {
                deepest_idx = i;
            }
            gap.max_clearance = std::max(gap.max_clearance, clearance);
        }

        gap.mean_clearance = count > 0 ? clearance_sum / static_cast<double>(count) : 0.0;
        gap.mean_clipped_clearance = count > 0
            ? clipped_clearance_sum / static_cast<double>(count)
            : 0.0;
        gap.weighted_center_angle = weight_sum > 1e-12
            ? weighted_sum / weight_sum
            : (gap.start_angle + gap.end_angle) / 2.0;
        gap.deepest_angle = scan.angles[deepest_idx];
        const double representative =
            0.5 * gap.weighted_center_angle + 0.5 * gap.deepest_angle;
        gap.score = std::pow(
                std::max(gap.mean_clipped_clearance, 0.0),
                config_.score_power)
            * gap.angular_width
            * std::exp(-config_.heading_weight * std::abs(representative));
        gaps.push_back(gap);
    };

    bool in_gap = false;
    size_t start_idx = 0;
    for (size_t i = 0; i <= n; ++i) {
        const bool usable = i < n
            && scan.valid[i]
            && scan.angles[i] >= lidar_config.angle_min
            && scan.angles[i] <= lidar_config.angle_max
            && eff_clearance[i] >= config_.min_score_range;

        if (usable && !in_gap) {
            start_idx = i;
            in_gap = true;
        } else if (!usable && in_gap) {
            append_gap(start_idx, i - 1);
            in_gap = false;
        }
    }
    return gaps;
}

DrivableGap FollowTheGap::selectDrivableGap(const std::vector<DrivableGap>& gaps) {
    if (gaps.empty()) return DrivableGap();

    const DrivableGap* best = &gaps.front();
    for (const auto& gap : gaps) {
        if (gap.score > best->score
            || (gap.score == best->score
                && std::abs(gap.weighted_center_angle)
                    < std::abs(best->weighted_center_angle))) {
            best = &gap;
        }
    }

    const DrivableGap* incumbent = nullptr;
    double incumbent_distance = std::numeric_limits<double>::infinity();
    if (has_selected_gap_) {
        for (const auto& gap : gaps) {
            const double distance = std::abs(gap.weighted_center_angle - previous_gap_angle_);
            if (distance < incumbent_distance) {
                incumbent_distance = distance;
                incumbent = &gap;
            }
        }
        // A gap farther than this is a different local branch, not the same
        // corridor moving slightly due to scan noise.
        if (incumbent_distance > 0.35) incumbent = nullptr;
    }

    const bool substantially_different = has_selected_gap_
        && std::abs(best->weighted_center_angle - previous_gap_angle_) > 0.35;
    const DrivableGap* selected = best;
    if (substantially_different && incumbent != nullptr) {
        const double incumbent_score = incumbent->score;
        const bool strong_alternative = best->score
            > incumbent_score * (1.0 + std::max(0.0, config_.gap_switch_margin));
        if (strong_alternative) {
            ++pending_alternative_count_;
        } else {
            pending_alternative_count_ = 0;
        }
        const int required_scans = std::max(1, config_.gap_switch_scans);
        if (pending_alternative_count_ < required_scans) {
            selected = incumbent;
        } else {
            pending_alternative_count_ = 0;
        }
    } else {
        pending_alternative_count_ = 0;
    }

    has_selected_gap_ = true;
    previous_gap_angle_ = selected->weighted_center_angle;
    previous_gap_score_ = selected->score;
    return *selected;
}

TargetResult FollowTheGap::computeTargetAngle(const DrivableGap& gap) const {
    TargetResult result;
    if (gap.angular_width < config_.min_gap_width) return result;

    // In a wide hairpin entrance the deepest return can belong to the dead
    // corner while the gap centre already points toward the continuation.
    // Averaging opposing, shallow directions would cancel them into a
    // straight command. Preserve the centre direction in that case; retain
    // the normal deepest-point blend for a decisive turn.
    const bool opposing_directions =
        gap.weighted_center_angle * gap.deepest_angle < 0.0;
    const bool shallow_deepest_return = std::abs(gap.deepest_angle) < 0.30;
    // Keep even a small non-zero centre direction: at a deep hairpin the
    // deepest return can oppose it and cancelling both into zero sends the
    // car straight into the corner.
    const bool meaningful_centre_direction =
        std::abs(gap.weighted_center_angle) > 0.02;
    const bool wide_gap = gap.angular_width > 2.5;
    const double target = opposing_directions
        && shallow_deepest_return
        && meaningful_centre_direction
        && wide_gap
        ? gap.weighted_center_angle
        : 0.5 * gap.weighted_center_angle + 0.5 * gap.deepest_angle;
    result.valid = true;
    result.angle = std::clamp(target, gap.start_angle, gap.end_angle);
    return result;
}

// =====================================================================
// Control
// =====================================================================

double FollowTheGap::calculateSpeed(double forward_clearance, double steering_angle) {
    // Range factor: how far ahead is clear
    double range_factor = std::clamp(forward_clearance / config_.speed_full_range, 0.0, 1.0);

    // Steering factor: slow down when turning
    double abs_steer = std::abs(steering_angle);
    double steer_factor = 1.0 - config_.steer_slowdown_gain * (abs_steer / config_.max_steering);
    steer_factor = std::clamp(steer_factor, 0.3, 1.0);

    double speed = config_.min_speed + (config_.max_speed - config_.min_speed) * range_factor * steer_factor;
    return std::clamp(speed, config_.min_speed, config_.max_speed);
}

double FollowTheGap::rateLimitSteering(double target, double last, double dt) {
    // Calculate maximum allowed change based on configured max steering rate
    // Might be redundant when driving on a real car since the physical 
    // steering mechanism already has rate limits,
    double max_change = config_.max_steering_rate * dt;
    double delta = target - last;
    if (std::abs(delta) > max_change) {
        return last + ((delta > 0) ? max_change : -max_change);
    }
    return target;
}

}  // namespace f1tenth_control

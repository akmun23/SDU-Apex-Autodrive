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
    last_single_gap_angle_ = 0.0;
    opposite_gap_cycles_ = 0;
    centered_gap_cycles_ = 0;
    has_single_gap_target_ = false;
    recovery_steering_sign_ = 0.0;
    recovery_opposite_cycles_ = 0;
    recovery_clear_cycles_ = 0;
    recovery_outside_sector_cycles_ = 0;
    last_exploration_overlap_ = 0.0;
    last_exploration_left_overlap_ = 0.0;
    last_exploration_right_overlap_ = 0.0;
    if (config_.exploration_history_enabled) {
        std::lock_guard<std::mutex> lock(exploration_mutex_);
        exploration_history_.clear();
        exploration_pose_valid_ = false;
        exploration_distance_since_sample_ = 0.0;
    }
}

void FollowTheGap::updateExplorationPose(double x, double y, double yaw) {
    if (!config_.exploration_history_enabled ||
        !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw)) {
        return;
    }

    std::lock_guard<std::mutex> lock(exploration_mutex_);
    const Point2D pose(x, y);
    if (!exploration_pose_valid_) {
        exploration_history_.push_back(pose);
        exploration_pose_valid_ = true;
        exploration_distance_since_sample_ = 0.0;
    } else {
        const Point2D previous(exploration_pose_.x, exploration_pose_.y);
        const double step = (pose - previous).norm();
        // A large jump is a reset/collision discontinuity, not trajectory.
        // Do not let it create a false explored branch.
        if (step > 1.0) {
            exploration_distance_since_sample_ = 0.0;
        } else if (step > 0.0) {
            // Odom is often faster than the vehicle moves during mapping.
            // Accumulate the sub-centimetre callbacks instead of discarding
            // them, otherwise the history contains only the initial pose and
            // cannot reject a later retrace.
            exploration_distance_since_sample_ += step;
            constexpr double kHistorySampleDistance = 0.02;
            if (exploration_distance_since_sample_ >=
                kHistorySampleDistance) {
                exploration_history_.push_back(pose);
                exploration_distance_since_sample_ = 0.0;
            }
        }
    }
    exploration_pose_ = ExplorationPose{x, y, yaw};
    constexpr std::size_t kMaximumHistoryPoints = 12000;
    while (exploration_history_.size() > kMaximumHistoryPoints) {
        exploration_history_.pop_front();
    }
}

// =====================================================================
// Main compute
// =====================================================================

FTGOutput FollowTheGap::compute(
    const std::vector<float>& ranges,
    double angle_min,
    double angle_max,
    double angle_increment
) {
    FTGOutput output;

    // --- Time step ---
    auto now = std::chrono::steady_clock::now();
    double dt = 0.025;  // default ~40 Hz

    // Calculate actual dt on subsequent calls for time-based rate limiting
    if (!first_compute_) {
        dt = std::chrono::duration<double>(now - last_compute_time_).count();
        dt = std::clamp(dt, 0.001, 0.5);
    }
    last_compute_time_ = now;

    // --- Handle empty scan ---
    if (ranges.empty()) {
        output.emergency_stop = true;
        output.command = DriveCommand(0.0, 0.0);
        return output;
    }

    // --- Step 1: Generic LiDAR preprocessing (median filter, range clip) ---
    ProcessedScan scan = lidar_processor_.processScan(
        ranges, angle_min, angle_max, angle_increment
    );

    // If no valid points after preprocessing, trigger emergency stop
    if (scan.filtered_ranges.empty()) {
        output.emergency_stop = true;
        output.command = DriveCommand(0.0, 0.0);
        return output;
    }

    // Initialise blocking flags (for visualisation)
    scan.disparity_blocked.assign(scan.filtered_ranges.size(), false);
    scan.bubble_blocked.assign(scan.filtered_ranges.size(), false);

    // --- Step 2: Closest-point detection on raw physical ranges ---
    // Wall margin is scoring inflation, not physical distance. Applying it
    // before this check makes the emergency threshold double-count margin and
    // can stop beside a safely cleared wall.
    output.closest_point_idx = lidar_processor_.findClosestPoint(scan);
    output.closest_point_dist = scan.filtered_ranges[output.closest_point_idx];

    // The absolute radial threshold is the hard stop.  A rectangular-footprint
    // margin violation above that threshold is recoverable: stopping there can
    // trap the vehicle beside a wall because it loses the forward motion
    // needed to steer away.  Keep the distinction explicit so the caller can
    // reduce speed without disabling the hard obstacle stop.
    output.footprint_clearance_limited = violatesFootprintClearance(scan);
    if (output.closest_point_dist < config_.emergency_brake_distance) {
        output.emergency_stop = true;
        // Brake longitudinally, but keep the steering direction that opens
        // the swept envelope.  A neutral steering command at the hard-stop
        // threshold leaves the car pointed at the obstacle and is especially
        // harmful in the mapping profile, where a close inner wall can still
        // be cleared by rotating in place.
        const double closest_angle = scan.angles[output.closest_point_idx];
        double escape_steering = 0.0;
        if (config_.emergency_prefer_last_steering &&
            std::abs(last_steering_) > 0.05 &&
            std::abs(closest_angle) > 0.15) {
            // At a circular-track inside corner, the nearest side wall can
            // be the wall the vehicle must turn around. Reversing away from
            // it at the hard threshold can send the car into the outside
            // wall. Preserve the already committed free-space turn when the
            // mapping profile explicitly requests it.
            escape_steering = last_steering_;
        } else if (std::abs(closest_angle) > 0.15) {
            escape_steering = closest_angle > 0.0 ?
                -config_.max_steering : config_.max_steering;
        } else if (std::abs(config_.ambiguous_front_recovery_sign) > 0.5) {
            escape_steering = config_.ambiguous_front_recovery_sign > 0.0 ?
                config_.max_steering : -config_.max_steering;
        } else if (std::abs(last_steering_) > 1.0e-6) {
            escape_steering = last_steering_;
        }
        output.command = DriveCommand(
            config_.emergency_rolling_speed, escape_steering);
        output.processed_scan = scan;

        // Still populate gaps for viz
        output.all_gaps = findGapsForViz(scan);
        return output;
    }

    // --- Step 3: Wall margin for path scoring ---
    applyWallMargin(scan);

    // --- Step 4: Disparity extension ---
    applyDisparityExtension(scan);

    // --- Step 5: Compute effective clearance per beam ---
    std::vector<double> eff_clearance = computeEffectiveClearance(scan);

    // --- Step 6: Compute weighted-centroid target angle ---
    double target_angle = computeTargetAngle(scan, eff_clearance);
    output.target_angle = target_angle;
    output.exploration_overlap = last_exploration_overlap_;

    // --- Step 7: EMA smoothing on the target angle ---
    // Not applied on first compute
    // Then creates a smoothing effect that helps prevent 
    // oscillations when target angle changes rapidly between 
    // different directions.
    if (first_compute_) {
        smoothed_target_ = target_angle;
        first_compute_ = false;
    } else {
        smoothed_target_ = config_.target_ema_alpha * target_angle + (1.0 - config_.target_ema_alpha) * smoothed_target_;
    }

    // Find raw steering command from smoothed target, applying gain and saturation
    double raw_steering = std::clamp(
        config_.steering_gain * smoothed_target_,
        -config_.max_steering,
        config_.max_steering
    );
    // Preserve the normal FTG decision before any side-recovery shaping.
    // This is the local branch indication available when both side sectors
    // look nearly symmetric at a closed-track corner.
    const double selected_gap_steering = raw_steering;
    const double closest_angle = scan.angles[output.closest_point_idx];
    double left_side_min = std::numeric_limits<double>::infinity();
    double right_side_min = std::numeric_limits<double>::infinity();
    for (size_t beam = 0; beam < scan.filtered_ranges.size(); ++beam) {
        const double angle = scan.angles[beam];
        if (!scan.valid[beam] || angle < 0.18 ||
            angle > config_.side_recovery_max_angle) {
            continue;
        }
        left_side_min = std::min(left_side_min, scan.filtered_ranges[beam]);
    }
    for (size_t beam = 0; beam < scan.filtered_ranges.size(); ++beam) {
        const double angle = scan.angles[beam];
        if (!scan.valid[beam] || angle > -0.18 ||
            angle < -config_.side_recovery_max_angle) {
            continue;
        }
        right_side_min = std::min(right_side_min, scan.filtered_ranges[beam]);
    }
    output.recovery_left_clearance = std::isfinite(left_side_min) ?
        left_side_min : 0.0;
    output.recovery_right_clearance = std::isfinite(right_side_min) ?
        right_side_min : 0.0;
    // The raw simulator scan extends to +/-135 deg, while the normal FTG
    // validity window is deliberately limited to the front +/-90 deg. Keep
    // rear context out of that normal mask: a rear return must never become a
    // forward target or an emergency obstacle. It is only a branch cue for
    // mapping at a genuinely front-facing closed corner. Read it from the
    // original scan rather than ``scan.valid``; otherwise the configured rear
    // sectors are always invalidated by LidarProcessor before they reach this
    // code and both rear clearances incorrectly remain zero.
    double rear_left_min = std::numeric_limits<double>::infinity();
    double rear_right_min = std::numeric_limits<double>::infinity();
    if (config_.use_rear_context) {
        const double minimum_rear_range = std::max(
            config_.lidar_config.range_min, config_.rear_context_min_range);
        for (size_t beam = 0; beam < ranges.size(); ++beam) {
            const double angle = angle_min +
                static_cast<double>(beam) * angle_increment;
            const double range = static_cast<double>(ranges[beam]);
            if (!std::isfinite(range) ||
                range < minimum_rear_range ||
                range > config_.lidar_config.range_max) {
                continue;
            }
            if (angle >= config_.rear_context_min_angle &&
                angle <= config_.rear_context_max_angle) {
                rear_left_min = std::min(rear_left_min, range);
            } else if (angle <= -config_.rear_context_min_angle &&
                       angle >= -config_.rear_context_max_angle) {
                rear_right_min = std::min(rear_right_min, range);
            }
        }
    }
    output.recovery_rear_left_clearance = std::isfinite(rear_left_min) ?
        rear_left_min : 0.0;
    output.recovery_rear_right_clearance = std::isfinite(rear_right_min) ?
        rear_right_min : 0.0;
    constexpr double kRecoverySideAdvantage = 0.03;
    double desired_recovery_sign = 0.0;
    constexpr double kNearFrontAngle = 0.15;
    const bool closest_in_recovery_sector =
        std::abs(closest_angle) <= config_.side_recovery_front_angle;
    const bool have_both_side_clearances =
        std::isfinite(left_side_min) && std::isfinite(right_side_min);
    const bool have_both_rear_clearances =
        std::isfinite(rear_left_min) && std::isfinite(rear_right_min);
    const bool rear_left_is_open =
        have_both_rear_clearances &&
        rear_left_min > rear_right_min + config_.rear_context_min_advantage;
    const bool rear_right_is_open =
        have_both_rear_clearances &&
        rear_right_min > rear_left_min + config_.rear_context_min_advantage;
    const bool rear_context_is_ambiguous =
        config_.use_rear_context && closest_in_recovery_sector &&
        have_both_rear_clearances && !rear_left_is_open && !rear_right_is_open;
    bool recovery_sign_evidence_strong = false;
    // A closed mapping track can present a symmetric front wall at a
    // hairpin: both local side sectors look open even though only one branch
    // continues the loop.  In that case a purely local nearest-beam rule has
    // no valid way to choose the branch.  A non-zero configured sign is an
    // explicit mapping-only loop-direction commitment; the default zero
    // keeps normal FTG behaviour unchanged.
    const bool side_context_is_ambiguous =
        have_both_side_clearances &&
        std::abs(left_side_min - right_side_min) <=
        kRecoverySideAdvantage;
    const bool selected_gap_is_ambiguous =
        std::abs(selected_gap_steering) <= 0.12;
    const bool configured_loop_direction =
        std::abs(config_.ambiguous_front_recovery_sign) > 0.5 &&
        closest_in_recovery_sector &&
        (rear_context_is_ambiguous ||
         (side_context_is_ambiguous && selected_gap_is_ambiguous));
    // If both local branches look plausible, compare their projected arcs
    // against the older odometry trail.  A single endpoint can miss a
    // retrace because the branches initially diverge and only merge farther
    // around the hairpin.  The overlap is already used as a soft candidate
    // score in computeTargetAngle(); keep it diagnostic here, but do not let
    // history directly command a left/right recovery sign. The scan must
    // remain free to select a genuinely open gap at each corner.
    if (config_.exploration_history_enabled &&
        closest_in_recovery_sector) {
        const double left_overlap = explorationOverlap(
            0.85, config_.exploration_probe_distance);
        const double right_overlap = explorationOverlap(
            -0.85, config_.exploration_probe_distance);
        last_exploration_left_overlap_ = left_overlap;
        last_exploration_right_overlap_ = right_overlap;
    } else {
        last_exploration_left_overlap_ = 0.0;
        last_exploration_right_overlap_ = 0.0;
    }
    // Prefer the explicit loop direction at an ambiguous front wall.  When
    // it is not configured, prefer the side with the larger measured opening.
    // The nearest beam
    // alone is not sufficient at an inner corner: the scan can alternate
    // between the two walls while the safe route remains on one side.
    if (configured_loop_direction) {
        desired_recovery_sign =
            config_.ambiguous_front_recovery_sign > 0.0 ? 1.0 : -1.0;
    } else if (config_.use_rear_context && closest_in_recovery_sector &&
               (rear_left_is_open || rear_right_is_open)) {
        // Positive scan angles are the vehicle's left side.  Select the side
        // with the larger rear-side clearance: that is the branch which has
        // space to continue around a U-turn.  This is deliberately evaluated
        // before the local nearest-beam heuristic, which was observed to
        // choose the wrong branch after the vehicle entered the corner.
        desired_recovery_sign = rear_left_is_open ? 1.0 : -1.0;
        recovery_sign_evidence_strong = true;
    } else if (!rear_context_is_ambiguous &&
               closest_in_recovery_sector && have_both_side_clearances &&
        right_side_min + kRecoverySideAdvantage < left_side_min) {
        desired_recovery_sign = 1.0;
        recovery_sign_evidence_strong = true;
    } else if (!rear_context_is_ambiguous &&
               closest_in_recovery_sector && have_both_side_clearances &&
               left_side_min + kRecoverySideAdvantage < right_side_min) {
        desired_recovery_sign = -1.0;
        recovery_sign_evidence_strong = true;
    } else if (closest_in_recovery_sector &&
               std::abs(selected_gap_steering) > 0.12) {
        // A normal gap direction can help initialize recovery, but it must
        // not reverse an already committed wall turn: the broad opening on
        // the outside of a hairpin is often free space while the track
        // continues around the inside wall.  Keep this as weak evidence; the
        // existing recovery latch will require stronger side/rear evidence or
        // clearance before changing sign.
        desired_recovery_sign = selected_gap_steering > 0.0 ? 1.0 : -1.0;
    } else if (closest_angle > kNearFrontAngle &&
               closest_angle <= config_.side_recovery_max_angle) {
        desired_recovery_sign = -1.0;
    } else if (closest_angle < -kNearFrontAngle &&
               closest_angle >= -config_.side_recovery_max_angle) {
        desired_recovery_sign = 1.0;
    } else if (std::abs(closest_angle) <= kNearFrontAngle) {
        // The closest beam can enter the front sector while the car is
        // rotating around an inner corner. Use the side-sector clearance to
        // retain the physically correct turn-away direction instead of
        // carrying a stale latch through the corner.
        const bool sides_are_ambiguous =
            !have_both_side_clearances ||
            std::abs(left_side_min - right_side_min) <=
            kRecoverySideAdvantage;
        if (sides_are_ambiguous &&
            std::abs(config_.ambiguous_front_recovery_sign) > 0.5) {
            desired_recovery_sign =
                config_.ambiguous_front_recovery_sign > 0.0 ? 1.0 : -1.0;
        } else if (std::isfinite(left_side_min) && std::isfinite(right_side_min) &&
            right_side_min + kRecoverySideAdvantage < left_side_min) {
            desired_recovery_sign = 1.0;
        } else if (std::isfinite(left_side_min) &&
                   std::isfinite(right_side_min) &&
                   left_side_min + kRecoverySideAdvantage < right_side_min) {
            desired_recovery_sign = -1.0;
        }
    }
    output.exploration_left_overlap = last_exploration_left_overlap_;
    output.exploration_right_overlap = last_exploration_right_overlap_;
    const bool has_front_side_obstacle =
        output.closest_point_dist < config_.side_recovery_distance &&
        desired_recovery_sign != 0.0 && closest_in_recovery_sector;
    const double recovery_release_margin = std::max(
        0.15, 0.20 * config_.side_recovery_full_steering_distance);
    const double recovery_release_distance =
        config_.side_recovery_distance + recovery_release_margin;
    const bool clear_recovery_candidate =
        output.closest_point_dist >= recovery_release_distance &&
        !output.footprint_clearance_limited;
    if (recovery_steering_sign_ != 0.0) {
        if (clear_recovery_candidate) {
            ++recovery_clear_cycles_;
            if (recovery_clear_cycles_ >=
                std::max(1, config_.recovery_clear_confirm_cycles)) {
                recovery_steering_sign_ = 0.0;
                recovery_opposite_cycles_ = 0;
                recovery_clear_cycles_ = 0;
            }
        } else {
            recovery_clear_cycles_ = 0;
        }
    }

    // Recovery is a front-side turn-away intervention. Once the nearest
    // return has moved outside that sector, a distance-only latch can become
    // actively wrong: the car has passed the corner, but the old sign keeps
    // steering toward the wall now visible at the side/rear. Mapping uses an
    // unlockable latch, so release it at this geometric transition and let
    // the selected gap take over. A locked configuration retains the stronger
    // commitment for workflows that explicitly require it.
    const bool clearly_past_recovery_sector =
        std::abs(closest_angle) > config_.side_recovery_front_angle + 0.04;
    if (recovery_steering_sign_ != 0.0 &&
        clearly_past_recovery_sector &&
        !output.footprint_clearance_limited) {
        ++recovery_outside_sector_cycles_;
    } else {
        recovery_outside_sector_cycles_ = 0;
    }
    if (recovery_steering_sign_ != 0.0 &&
        recovery_outside_sector_cycles_ >=
        std::max(1, config_.recovery_clear_confirm_cycles) &&
        !output.footprint_clearance_limited) {
        recovery_steering_sign_ = 0.0;
        recovery_opposite_cycles_ = 0;
        recovery_clear_cycles_ = 0;
        recovery_outside_sector_cycles_ = 0;
    }

    if (has_front_side_obstacle) {
        recovery_clear_cycles_ = 0;
        const double desired_sign = desired_recovery_sign;
        if (recovery_steering_sign_ == 0.0) {
            recovery_steering_sign_ = desired_sign;
            recovery_opposite_cycles_ = 0;
        } else if (desired_sign != recovery_steering_sign_) {
            // A real inner-corner transition can move the closest beam from
            // one side of the front sector to the other before the old wall
            // has reached the release distance. Require a few consecutive
            // opposite-side scans, rather than holding the stale direction
            // or reversing on one noisy beam.
            ++recovery_opposite_cycles_;
            const bool may_switch_side =
                !config_.lock_recovery_side_until_clear &&
                recovery_sign_evidence_strong;
            if (may_switch_side &&
                recovery_opposite_cycles_ >=
                std::max(1, config_.recovery_switch_confirm_cycles)) {
                recovery_steering_sign_ = desired_sign;
                recovery_opposite_cycles_ = 0;
            }
        } else {
            recovery_opposite_cycles_ = 0;
        }
    } else {
        recovery_opposite_cycles_ = 0;
    }
    output.recovery_steering_sign = recovery_steering_sign_;
    // Once selected, keep the turn-away command authoritative until the wall
    // has cleared the hysteresis envelope. A single scan whose closest beam
    // lands just outside the sector must not let a normal gap command fight
    // the recovery turn.
    const bool carry_recovery =
        recovery_steering_sign_ != 0.0 &&
        (output.closest_point_dist < recovery_release_distance ||
         recovery_clear_cycles_ <
         std::max(1, config_.recovery_clear_confirm_cycles));
    if (carry_recovery) {
        // A close front-side wall is more reliable than a branch choice made
        // from a rapidly changing gap profile. Use a proportional turn-away
        // command while there is room, reserving full steering for the
        // emergency envelope. This prevents alternating side detections from
        // producing a full left/right oscillation in a narrow corridor.
        const double full_distance = std::min(
            config_.side_recovery_full_steering_distance,
            config_.side_recovery_distance - 1.0e-3);
        const double recovery_span = std::max(
            config_.side_recovery_distance - full_distance, 1.0e-3);
        const double proximity = std::clamp(
            (config_.side_recovery_distance - output.closest_point_dist) /
            recovery_span, 0.0, 1.0);
        double recovery_magnitude =
            config_.side_recovery_min_steering +
            proximity * (config_.max_steering - config_.side_recovery_min_steering);
        const double away_sign = recovery_steering_sign_;
        // If the selected gap already points away from the close wall, retain
        // that stronger command. Only replace a command that points into the
        // wall with the proportional recovery authority.
        if (raw_steering * away_sign > 0.0) {
            recovery_magnitude = std::max(
                recovery_magnitude, std::abs(raw_steering));
        }
        if (config_.recovery_override_selected_gap) {
            const double recovery_target = away_sign * recovery_magnitude;
            double recovery_blend = config_.recovery_gap_blend;
            if (raw_steering * away_sign < 0.0 &&
                output.footprint_clearance_limited) {
                // Keep the selected gap authoritative while the vehicle is
                // still outside its footprint envelope.  The old rule raised
                // this blend to one solely from distance, so a valid gap was
                // replaced by saturated recovery steering well before a
                // collision was geometrically imminent.  Only an actual
                // footprint-margin violation may increase recovery authority;
                // the emergency-distance guard remains a separate hard stop.
                const double conflict_proximity = std::clamp(
                    (config_.side_recovery_distance -
                     output.closest_point_dist) / recovery_span,
                    0.0, 1.0);
                recovery_blend = std::max(
                    recovery_blend, conflict_proximity);
            }
            raw_steering =
                (1.0 - recovery_blend) * raw_steering +
                recovery_blend * recovery_target;
        }
        output.side_recovery = true;
    }
    output.raw_steering = raw_steering;

    // --- Step 8: Time-based steering rate limiting ---
    double steering = rateLimitSteering(raw_steering, last_steering_, dt);
    const bool recovery_requires_sign_change =
        carry_recovery &&
        last_steering_ * recovery_steering_sign_ < 0.0;
    const bool close_front_corner_recovery =
        carry_recovery && std::abs(closest_angle) <= 0.35;
    if (recovery_requires_sign_change || close_front_corner_recovery) {
        // A recovery direction is a collision-avoidance intervention. Do not
        // spend the first several cycles slewing through the old turn sign;
        // that leaves the car pointed at the newly detected front corner. The
        // close-front case also bypasses slew whenever a corner is already in
        // the central swept envelope, even if the previous command happened
        // to have the same sign after a noisy side transition.
        steering = raw_steering;
    } else if (carry_recovery && steering * recovery_steering_sign_ <= 0.0) {
        // Never publish a rate-limited command that turns into the obstacle.
        // A recovery command may need to cross through zero from the previous
        // normal-following command, but holding the old sign while the wall is
        // inside the swept envelope is unsafe.  Preserve the recovery
        // direction immediately; the next cycles can resume normal slew
        // limiting once the command has the correct sign.
        steering = raw_steering;
    }
    last_steering_ = steering;

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
    // If no valid beams in forward cone, assume some small clearance to avoid zero speed
    fwd_clearance = (fwd_count > 0) ? fwd_clearance / fwd_count : 0.5;

    // Calculate speed using the steering demand that will be reached, not
    // only the currently slew-limited command.  At the first scan of a turn
    // the actuator may still be near zero while raw_steering is already at
    // the corner limit; using only ``steering`` then commands high speed
    // into a turn before the steering rate limiter catches up.
    const double speed_steering =
        std::max(std::abs(steering), std::abs(raw_steering));
    double speed = calculateSpeed(fwd_clearance, speed_steering);
    if (output.footprint_clearance_limited) {
        // A footprint-margin violation is close enough that rolling speed must
        // be limited while the steering command clears the swept envelope.
        // Do not apply this cap merely because side recovery is active: a
        // side/rear return is also the normal way to leave a wall-following
        // sector, and forcing minimum speed there prevents FTG from reaching
        // the next gap.  The absolute emergency threshold above still
        // commands zero.
        speed = std::min(speed, config_.min_speed);
    }
    output.command = DriveCommand(speed, steering);

    // --- Step 10: Populate gaps for visualisation ---
    output.all_gaps = findGapsForViz(scan);
    output.selected_gap = findBestGapForViz(output.all_gaps);
    output.processed_scan = scan;

    return output;
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

bool FollowTheGap::violatesFootprintClearance(
    const ProcessedScan& scan) const {
    const double half_width = std::max(
        0.0,
        config_.car_width * 0.5 +
        std::max(0.0, config_.side_safety_margin) +
        std::max(0.0, config_.virtual_width_inflation));
    const double front_extent = std::max(
        0.0,
        config_.car_length - config_.rear_overhang - config_.lidar_offset_x +
        std::max(0.0, config_.virtual_front_inflation));
    const double rear_extent = std::max(
        0.0,
        config_.rear_overhang + config_.lidar_offset_x +
        std::max(0.0, config_.virtual_rear_inflation));
    const double guard = std::max(0.0, config_.footprint_clearance);

    for (size_t i = 0; i < scan.filtered_ranges.size(); ++i) {
        if (!scan.valid[i]) {
            continue;
        }

        const double cosine = std::cos(scan.angles[i]);
        const double sine = std::sin(scan.angles[i]);
        const double longitudinal_extent =
            cosine >= 0.0 ? front_extent : rear_extent;
        const double longitudinal_distance =
            std::abs(cosine) > 1e-9
                ? longitudinal_extent / std::abs(cosine)
                : std::numeric_limits<double>::infinity();
        const double lateral_distance =
            std::abs(sine) > 1e-9
                ? half_width / std::abs(sine)
                : std::numeric_limits<double>::infinity();
        const double body_distance = std::min(
            longitudinal_distance, lateral_distance);
        const double required_distance = std::max(
            config_.emergency_brake_distance, body_distance + guard);

        if (scan.filtered_ranges[i] < required_distance) {
            return true;
        }
    }

    return false;
}

void FollowTheGap::applyDisparityExtension(ProcessedScan& scan) {
    // If scan has fewer than 2 beams, skip disparity extension
    if (scan.filtered_ranges.size() < 2) return;

    // For convenience, create a reference to the filtered ranges vector
    std::vector<double>& ranges = scan.filtered_ranges;
    const double half_car = config_.car_width / 2.0 +
        std::max(0.0, config_.side_safety_margin) +
        std::max(0.0, config_.virtual_width_inflation);
    const auto& lidar_config = lidar_processor_.getConfig();

    // Cap pathological extensions while allowing the full body plus explicit
    // side margin to block unsafe directions around wall edges.
    const int max_extension = static_cast<int>(
        std::ceil(
            std::max(0.0, config_.max_disparity_extension_angle) /
            std::abs(scan.angle_increment))
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
    const double half_car = (
        config_.car_width / 2.0 +
        std::max(0.0, config_.side_safety_margin) +
        std::max(0.0, config_.virtual_width_inflation)
    ) * config_.clearance_cone_scale;
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

double FollowTheGap::computeTargetAngle(
    const ProcessedScan& scan,
    const std::vector<double>& eff_clearance
) {
    last_exploration_overlap_ = 0.0;
    if (config_.select_single_gap) {
        // A global weighted centroid is unsafe in a hairpin: two valid
        // openings on opposite sides can average to a straight command even
        // though the straight beam is already blocked. Select one contiguous
        // opening and aim at its clearance-weighted centre instead.
        double best_score = -std::numeric_limits<double>::infinity();
        double best_angle = 0.0;
        size_t i = 0;
        const auto& lidar_config = lidar_processor_.getConfig();
        while (i < eff_clearance.size()) {
            while (i < eff_clearance.size() &&
                   (scan.angles[i] < lidar_config.angle_min ||
                    scan.angles[i] > lidar_config.angle_max ||
                    eff_clearance[i] < config_.min_score_range)) {
                ++i;
            }
            if (i >= eff_clearance.size()) {
                break;
            }

            const size_t start = i;
            double sum_weight = 0.0;
            double weighted_angle = 0.0;
            double max_clearance = 0.0;
            while (i < eff_clearance.size() &&
                   scan.angles[i] >= lidar_config.angle_min &&
                   scan.angles[i] <= lidar_config.angle_max &&
                   eff_clearance[i] >= config_.min_score_range) {
                const double weight = std::pow(
                    eff_clearance[i] - config_.min_score_range,
                    config_.score_power);
                sum_weight += weight;
                weighted_angle += scan.angles[i] * weight;
                max_clearance = std::max(max_clearance, eff_clearance[i]);
                ++i;
            }

            const size_t end = i - 1;
            const double angular_width =
                std::max(0.0, scan.angles[end] - scan.angles[start]);
            if (sum_weight <= 1.0e-9 ||
                angular_width < config_.min_gap_width) {
                continue;
            }

            const double center = weighted_angle / sum_weight;
            const double score =
                std::pow(max_clearance - config_.min_score_range,
                         config_.score_power) * angular_width *
                std::exp(-config_.heading_weight * std::abs(center));
            const double overlap = explorationOverlap(center, max_clearance);
            const double exploration_score = score * std::exp(
                -config_.exploration_branch_penalty * overlap);
            if (exploration_score > best_score) {
                best_score = exploration_score;
                best_angle = center;
                last_exploration_overlap_ = overlap;
            }
        }
        if (best_score > -std::numeric_limits<double>::infinity()) {
            if (config_.avoid_close_side) {
                double left_min = std::numeric_limits<double>::infinity();
                double right_min = std::numeric_limits<double>::infinity();
                for (size_t beam = 0; beam < eff_clearance.size(); ++beam) {
                    const double angle = scan.angles[beam];
                    if (angle >= 0.15 && angle <= 1.20 &&
                        eff_clearance[beam] >= config_.min_score_range) {
                        left_min = std::min(left_min, eff_clearance[beam]);
                    } else if (angle <= -0.15 && angle >= -1.20 &&
                               eff_clearance[beam] >= config_.min_score_range) {
                        right_min = std::min(right_min, eff_clearance[beam]);
                    }
                }
                constexpr double kSideClearanceAdvantage = 0.03;
                if (std::isfinite(left_min) && std::isfinite(right_min) &&
                    right_min + kSideClearanceAdvantage < left_min) {
                    // The negative-angle side is closer.  Turn away from it
                    // even when the selected gap is currently near zero;
                    // waiting for a large gap target leaves too little room
                    // for the vehicle to rotate before the footprint guard.
                    best_angle = std::max(
                        std::abs(best_angle), config_.close_side_turn_angle);
                } else if (std::isfinite(left_min) && std::isfinite(right_min) &&
                           left_min + kSideClearanceAdvantage < right_min) {
                    best_angle = -std::max(
                        std::abs(best_angle), config_.close_side_turn_angle);
                }
            }
            // Apply opposite-gap confirmation after side-safety shaping too.
            // Keep the pending confirmation alive through a brief centred
            // scan.  At an inner corner the selected gap can momentarily
            // collapse to near-zero between two opposite openings; resetting
            // the counter there permits an unsafe full reversal.
            // In a broad hairpin opening the weighted centroid can wander
            // through a few tenths of a radian while the car is still
            // committed to the same bend. Treat that small opposite target
            // as centred scan noise; a clearly opposite opening still passes
            // through the normal confirmation counter below.
            constexpr double kGapDirectionEpsilon = 0.50;
            if (std::abs(best_angle) > kGapDirectionEpsilon) {
                centered_gap_cycles_ = 0;
                if (has_single_gap_target_ &&
                    ((best_angle > 0.0) != (last_single_gap_angle_ > 0.0))) {
                    ++opposite_gap_cycles_;
                    if (opposite_gap_cycles_ <
                        std::max(1, config_.gap_switch_confirm_cycles)) {
                        best_angle = last_single_gap_angle_;
                    } else {
                        opposite_gap_cycles_ = 0;
                    }
                } else {
                    opposite_gap_cycles_ = 0;
                }
                last_single_gap_angle_ = best_angle;
                has_single_gap_target_ = true;
            } else if (has_single_gap_target_ &&
                       std::abs(last_single_gap_angle_) >
                       kGapDirectionEpsilon &&
                       centered_gap_cycles_ <
                       std::max(1, config_.gap_switch_confirm_cycles)) {
                // A hairpin can briefly look centred while the scan is
                // crossing from its entry wall to the selected continuation.
                // Preserve the already selected direction for a few cycles;
                // this prevents a transient centroid at zero from steering
                // into the opposite, retracing branch. A later non-centred
                // opposite gap still goes through the normal confirmation.
                ++centered_gap_cycles_;
                return last_single_gap_angle_;
            } else {
                centered_gap_cycles_ = 0;
                opposite_gap_cycles_ = std::max(0, opposite_gap_cycles_ - 1);
            }
            return best_angle;
        }
    }

    // Initialize accumulators for weighted average
    const size_t n = scan.filtered_ranges.size();
    const auto& lidar_config = lidar_processor_.getConfig();

    double sum_score = 0.0;
    double sum_weighted_angle = 0.0;

    // Loop through all beams and compute score based on effective clearance and heading
    for (size_t i = 0; i < n; ++i) {
        double angle = scan.angles[i];

        // Skip beams outside the configured angular processing range or invalid beams
        if (angle < lidar_config.angle_min || angle > lidar_config.angle_max) continue;

        double clearance = eff_clearance[i];
        // Skip beams that are too close to be considered drivable
        if (clearance < config_.min_score_range) continue;

        // Score = (clearance - min_score_range) ^ power  *  exp(-heading_weight * |angle|)
        double base = clearance - config_.min_score_range;
        double score = std::pow(base, config_.score_power)
                     * std::exp(-config_.heading_weight * std::abs(angle));

        // Accumulate weighted angle and total score
        sum_score += score;
        sum_weighted_angle += angle * score;
    }

    if (sum_score < 1e-9) {
        // No drivable direction found — default to straight ahead
        return 0.0;
    }

    return sum_weighted_angle / sum_score;
}

double FollowTheGap::explorationOverlap(
    double target_angle,
    double clearance) const {
    if (!config_.exploration_history_enabled) {
        return 0.0;
    }

    std::lock_guard<std::mutex> lock(exploration_mutex_);
    if (!exploration_pose_valid_ || exploration_history_.size() < 2) {
        return 0.0;
    }

    const double probe_distance = std::clamp(
        std::min(clearance, config_.exploration_probe_distance),
        1.0, config_.exploration_probe_distance);
    // Ignore the recent trajectory: the candidate is expected to overlap the
    // corridor the car is currently leaving during a normal bend. Only older
    // points are evidence that this branch would retrace an already explored
    // route.
    double distance_from_recent = 0.0;
    std::vector<Point2D> old_history;
    old_history.reserve(exploration_history_.size());
    for (std::size_t index = exploration_history_.size(); index-- > 0;) {
        if (index + 1 < exploration_history_.size()) {
            const Point2D delta = exploration_history_[index + 1] -
                exploration_history_[index];
            distance_from_recent += delta.norm();
        }
        if (distance_from_recent < config_.exploration_recent_exclusion_distance) {
            continue;
        }
        old_history.push_back(exploration_history_[index]);
    }

    if (old_history.empty() ||
        config_.exploration_history_radius <= 1.0e-6) {
        return 0.0;
    }

    // Score several points along the projected arc.  The maximum overlap is
    // the relevant value: a branch that merges into the old trail farther
    // around a U-turn is still a retrace candidate.
    double maximum_overlap = 0.0;
    constexpr int kArcSamples = 7;
    for (int sample = 1; sample <= kArcSamples; ++sample) {
        const double distance = probe_distance *
            static_cast<double>(sample) / static_cast<double>(kArcSamples);
        const Point2D probe(
            exploration_pose_.x + distance *
                std::cos(exploration_pose_.yaw + target_angle),
            exploration_pose_.y + distance *
                std::sin(exploration_pose_.yaw + target_angle));
        double nearest_old_distance = std::numeric_limits<double>::infinity();
        for (const auto & old_point : old_history) {
            nearest_old_distance = std::min(
                nearest_old_distance, (probe - old_point).norm());
        }
        maximum_overlap = std::max(
            maximum_overlap,
            std::exp(-0.5 * std::pow(
                nearest_old_distance / config_.exploration_history_radius, 2.0)));
    }
    return maximum_overlap;
}

// =====================================================================
// Gap detection (visualisation only)
// =====================================================================

std::vector<Gap> FollowTheGap::findGapsForViz(const ProcessedScan& scan) {
    std::vector<Gap> gaps;
    if (scan.filtered_ranges.empty()) return gaps;

    const auto& lidar_config = lidar_processor_.getConfig();
    bool in_gap = false;
    Gap current_gap;
    size_t current_gap_count = 0;

    for (size_t i = 0; i < scan.filtered_ranges.size(); ++i) {
        double range = scan.filtered_ranges[i];
        double angle = scan.angles[i];
        if (angle < lidar_config.angle_min || angle > lidar_config.angle_max) continue;

        bool is_gap = (range >= config_.gap_threshold && scan.valid[i]);

        if (is_gap && !in_gap) {
            in_gap = true;
            current_gap = Gap();
            current_gap.start_idx = i;
            current_gap.start_angle = angle;
            current_gap.min_range = range;
            current_gap.max_range = range;
            current_gap.deepest_idx = i;
            current_gap.deepest_range = range;
            current_gap.avg_range = range;
            current_gap_count = 1;
        } else if (is_gap && in_gap) {
            current_gap.min_range = std::min(current_gap.min_range, range);
            if (range > current_gap.deepest_range) {
                current_gap.deepest_range = range;
                current_gap.deepest_idx = i;
            }
            current_gap.max_range = std::max(current_gap.max_range, range);
            current_gap.avg_range += range;
            ++current_gap_count;
        } else if (!is_gap && in_gap) {
            in_gap = false;
            current_gap.end_idx = i - 1;
            current_gap.end_angle = scan.angles[i - 1];
            current_gap.angular_width = current_gap.end_angle - current_gap.start_angle;
            if (current_gap_count > 0) {
                current_gap.avg_range /= static_cast<double>(current_gap_count);
            }
            if (current_gap.angular_width >= config_.min_gap_width) {
                gaps.push_back(current_gap);
            }
        }
    }

    if (in_gap) {
        current_gap.end_idx = scan.filtered_ranges.size() - 1;
        current_gap.end_angle = scan.angles.back();
        current_gap.angular_width = current_gap.end_angle - current_gap.start_angle;
        if (current_gap_count > 0) {
            current_gap.avg_range /= static_cast<double>(current_gap_count);
        }
        if (current_gap.angular_width >= config_.min_gap_width) {
            gaps.push_back(current_gap);
        }
    }

    return gaps;
}

Gap FollowTheGap::findBestGapForViz(const std::vector<Gap>& gaps) {
    if (gaps.empty()) return Gap();

    // Pick widest gap that is closest to straight ahead
    double best = -std::numeric_limits<double>::infinity();
    const Gap* best_gap = &gaps[0];
    for (const auto& g : gaps) {
        double score = g.angular_width * g.deepest_range
                     * std::exp(-0.5 * std::abs(g.centerAngle()));
        if (score > best) {
            best = score;
            best_gap = &g;
        }
    }
    return *best_gap;
}

// =====================================================================
// Control
// =====================================================================

double FollowTheGap::calculateSpeed(
    double forward_clearance,
    double steering_angle) {
    // Range factor: how far ahead is clear
    double range_factor = std::clamp(forward_clearance / config_.speed_full_range, 0.0, 1.0);

    // Steering factor: slow down when turning
    double abs_steer = std::abs(steering_angle);
    double steer_factor = 1.0 - config_.steer_slowdown_gain * (abs_steer / config_.max_steering);
    steer_factor = std::clamp(steer_factor, 0.3, 1.0);

    double speed = config_.min_speed +
        (config_.max_speed - config_.min_speed) * range_factor * steer_factor;
    return std::clamp(speed, config_.min_speed, config_.max_speed);
}

double FollowTheGap::rateLimitSteering(double target, double last, double dt) {
    // Limit the discrete command step at the native 10 Hz interface.  This
    // keeps scan-to-scan steering changes bounded for the simulated actuator.
    double max_change = config_.max_steering_rate * dt;
    double delta = target - last;
    if (std::abs(delta) > max_change) {
        return last + ((delta > 0) ? max_change : -max_change);
    }
    return target;
}

}  // namespace f1tenth_control

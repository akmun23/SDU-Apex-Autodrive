#ifndef F1TENTH_CONTROL_FOLLOW_THE_GAP_HPP_
#define F1TENTH_CONTROL_FOLLOW_THE_GAP_HPP_

/**
 * @file follow_the_gap.hpp
 * @brief Weighted free-space Follow-The-Gap reactive steering controller.
 * @details Implements a reactive controller using LiDAR clearance
 *          scoring. Does not require a pre-planned trajectory. Intended for
 *          obstacle avoidance and gap-following in unknown or dynamic environments.
 *          Algorithm state: last_steering_, smoothed_target_, first_compute_,
 *          last_compute_time_. All LiDAR preprocessing is delegated to LidarProcessor.
 * @dependencies types.hpp, lidar_processor.hpp, <chrono>, <vector>
 */

#include "common/types.hpp"
#include "common/math_utils.hpp"
#include "common/lidar_processor.hpp"
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mutex>
#include <deque>

namespace f1tenth_control {

/**
 * @brief Configuration parameters for the weighted free-space FTG controller.
 *
 * The algorithm computes an effective clearance for each beam direction,
 * scores candidate directions, and selects steering by weighted centroiding.
 */
struct FTGConfig {
    // -- Vehicle parameters ---------------------------------------------------
    double wheelbase{0.324};         // Distance between axles (m)
    double car_length{0.500};        // Full bumper-to-bumper length (m)
    double car_width{0.270};         // Full body width (m)
    double rear_overhang{0.080};     // Rear axle to rear bumper (m)
    double lidar_offset_x{0.2733};   // Rear axle to LiDAR origin (m)
    // Conservative virtual footprint inflation used by obstacle processing.
    // These margins model the swept front/side envelope without changing the
    // physical actuator or vehicle model.
    double virtual_width_inflation{0.0}; // Added to each side (m)
    double virtual_front_inflation{0.0}; // Added ahead of LiDAR (m)
    double virtual_rear_inflation{0.0};  // Added behind LiDAR (m)

    // -- Speed control --------------------------------------------------------
    double max_speed{2.0};           // Maximum speed (m/s)
    double min_speed{1.0};           // Minimum speed (m/s)
    double speed_full_range{4.0};    // Range (m) at which full speed is used
    double steer_slowdown_gain{0.5}; // How much steering reduces speed (0-1)

    // -- Steering control -----------------------------------------------------
    double max_steering{0.5236};     // [rad] Maximum steering angle (~24 deg, calibrated)
    double steering_gain{1.0};       // Proportional gain on target angle
    double max_steering_rate{3.2};// [rad/s] Maximum steering change rate
    double target_ema_alpha{0.35};   // EMA smoothing for target angle (lower = smoother)

    // -- Weighted free-space scoring ------------------------------------------
    double heading_weight{1.0};      // Exponential decay for non-forward dirs
    double score_power{2.0};         // Raise effective clearance to this power
                                     // Squaring (power=2) emphasizes larger 
                                     // clearances more than smaller ones, 
                                     // creating sharper distinctions between 
                                     // good and bad directions.

    double clearance_cone_scale{1.5};// Multiplier on car half-width for cone
    double min_score_range{0.3};     // Beams shorter than this get zero score (m)
    // Select one contiguous free-space opening instead of averaging all
    // openings. This is useful for slow mapping in hairpins where a global
    // centroid can cancel symmetric left/right gaps and delay turn-in.
    bool select_single_gap{false};
    // In mapping mode, reject a selected target that points into a materially
    // closer side wall when the opposite side has a valid opening.
    bool avoid_close_side{false};
    double close_side_turn_angle{0.22};
    int gap_switch_confirm_cycles{6};

    // -- Safety ---------------------------------------------------------------
    double emergency_brake_distance{0.15}; // Brake if any beam closer (m)
    // Optional mapping-only crawl speed while holding escape steering at the
    // hard-stop threshold. Zero preserves a stationary emergency stop.
    double emergency_rolling_speed{0.0};
    double footprint_clearance{0.08}; // Extra free space beyond body (m)
    double side_recovery_distance{0.35}; // Begin turn-away recovery (m)
    double side_recovery_max_angle{1.20}; // Front-side sector for recovery (rad)
    // Only a close return in this narrower front sector may override the
    // selected gap. Returns farther toward the side are track boundaries and
    // must not force the car away from a valid bend.
    double side_recovery_front_angle{0.65};
    double side_recovery_min_steering{0.12}; // Gentle turn-away authority (rad)
    double side_recovery_full_steering_distance{0.30}; // Full authority below this (m)
    int recovery_switch_confirm_cycles{4}; // Opposite-side confirmation
    int recovery_clear_confirm_cycles{5}; // Clear scans before release
    bool lock_recovery_side_until_clear{false}; // At most one mapping switch
    // Mapping corridors may contain a close inside wall that the selected
    // free-space turn must follow. When false, side recovery only limits
    // speed and does not replace that selected turn.
    bool recovery_override_selected_gap{true};
    // Fraction of recovery authority blended into the normal gap command.
    // Zero leaves gap selection authoritative; one is a hard override.
    double recovery_gap_blend{1.0};
    // Optional mapping-only commitment for a genuinely front-facing obstacle
    // when both side sectors are equally open. Zero keeps the measured-side
    // heuristic; -1/+1 selects the known traversable branch.
    double ambiguous_front_recovery_sign{0.0};
    // Mapping-only branch cue from the rear side of the simulator's 270 deg
    // scan.  These beams are not used as a direct driving target or hard-stop
    // source; they only disambiguate which side of a hairpin remains open.
    bool use_rear_context{false};
    double rear_context_min_angle{1.57079632679};
    double rear_context_max_angle{2.35619449019};
    // Ignore invalid/min-range returns and the vehicle body in the rear cue.
    // They are not track-wall evidence and otherwise create false branch
    // closures when a rear beam contains the LiDAR minimum sentinel.
    double rear_context_min_range{0.20};
    double rear_context_min_advantage{0.20};
    // Mapping-only option to preserve a committed turn at a side corner.
    bool emergency_prefer_last_steering{false};

    // Mapping-only exploration memory.  A reactive scan cannot distinguish
    // an unexplored branch from the corridor the car just retraced.  When
    // enabled, contiguous gap candidates whose forward projection overlaps
    // older trajectory are penalized; the recent trajectory is excluded so
    // ordinary bends remain followable.
    bool exploration_history_enabled{false};
    double exploration_probe_distance{2.5};
    double exploration_recent_exclusion_distance{1.5};
    double exploration_history_radius{0.75};
    double exploration_branch_penalty{5.0};

    // -- LiDAR processing -----------------------------------------------------
    double disparity_threshold{0.5}; // Threshold for disparity extension (m)
    double wall_margin{0.15};        // Uniform obstacle inflation (m)
    double side_safety_margin{0.10}; // Extra margin beyond each body side (m)
    double max_disparity_extension_angle{0.7854}; // Maximum edge inflation (rad)
    double gap_threshold{0.5};       // Min range to count as "gap" in viz (m)
    double min_gap_width{0.15};      // Min angular width of gap for viz (rad)

    // -- Generic LiDAR preprocessing -----------------------------------------
    LidarProcessorConfig lidar_config;
};

/**
 * @brief Output of one FTG compute cycle.
 */
struct FTGOutput {
    DriveCommand command;            // Drive command generated for current cycle.
    Gap selected_gap;                // Best visualization gap selected from all_gaps.
    size_t closest_point_idx{0};     // Index of nearest detected point in processed scan.
    double closest_point_dist{0.0};  // Distance to nearest detected point.
    bool emergency_stop{false};      // True when emergency brake condition is active.
    bool footprint_clearance_limited{false};  // Tight body margin; slow recovery.
    bool side_recovery{false};       // Close front-side beam; steer away.
    double target_angle{0.0};        // Selected free-space target before gain.
    double raw_steering{0.0};        // Gain/saturation result before rate limiting.
    double recovery_steering_sign{0.0};
    double recovery_left_clearance{0.0};
    double recovery_right_clearance{0.0};
    double recovery_rear_left_clearance{0.0};
    double recovery_rear_right_clearance{0.0};
    double exploration_overlap{0.0};
    double exploration_left_overlap{0.0};
    double exploration_right_overlap{0.0};
    std::vector<Gap> all_gaps;       // All detected gaps used for diagnostics/visualization.
    ProcessedScan processed_scan;    // Fully processed scan used for command generation.
};

/**
 * @brief Weighted free-space Follow-The-Gap controller.
 * The controller: 
 * 1. Preprocesses LiDAR data. 
 * 2. Applies conservative safety shaping.
 * 3. Computes direction-wise drivability scores.
 * 4. Outputs bounded steering/speed commands.
 */
class FollowTheGap {
public:
    /**
     * @brief Construct a FollowTheGap controller instance.
     * @param config FTG configuration values used by the controller.
     */
    explicit FollowTheGap(const FTGConfig& config);

    /**
     * @brief Update runtime FTG configuration.
     * @param config New FTG configuration values.
     * @return None.
     */
    void setConfig(const FTGConfig& config);

    /**
     * @brief Get current FTG configuration.
     * @return Const reference to active configuration.
     */
    const FTGConfig& getConfig() const { return config_; }

    /**
     * @brief Compute FTG command output from one LiDAR scan.
     * @param ranges Raw LiDAR range vector.
     * @param angle_min Start angle of LiDAR scan.
     * @param angle_max End angle of LiDAR scan.
     * @param angle_increment Angular increment between neighboring beams.
     * @return FTGOutput with command and diagnostics for the current cycle.
     */
    FTGOutput compute(
        const std::vector<float>& ranges,
        double angle_min,
        double angle_max,
        double angle_increment
    );

    /**
     * @brief Update mapping-only pose history used to reject retraced gaps.
     * @param x Pose x coordinate in the odometry/world frame.
     * @param y Pose y coordinate in the odometry/world frame.
     * @param yaw Pose heading in the same frame.
     */
    void updateExplorationPose(double x, double y, double yaw);

    /**
     * @brief Get read-only access to shared LiDAR preprocessor.
     * @return Const reference to internal LidarProcessor.
     */
    const LidarProcessor& getLidarProcessor() const { return lidar_processor_; }

    /**
     * @brief Reset temporal controller state.
     * @return None.
     */
    void reset();

private:
    FTGConfig config_;
    LidarProcessor lidar_processor_;
    double last_steering_{0.0};
    double smoothed_target_{0.0};
    bool first_compute_{true};
    double last_single_gap_angle_{0.0};
    int opposite_gap_cycles_{0};
    int centered_gap_cycles_{0};
    bool has_single_gap_target_{false};
    double recovery_steering_sign_{0.0};
    int recovery_opposite_cycles_{0};
    int recovery_clear_cycles_{0};
    int recovery_outside_sector_cycles_{0};
    std::chrono::steady_clock::time_point last_compute_time_;

    struct ExplorationPose {
        double x{0.0};
        double y{0.0};
        double yaw{0.0};
    };
    mutable std::mutex exploration_mutex_;
    bool exploration_pose_valid_{false};
    ExplorationPose exploration_pose_{};
    std::deque<Point2D> exploration_history_;
    double exploration_distance_since_sample_{0.0};
    double last_exploration_overlap_{0.0};
    double last_exploration_left_overlap_{0.0};
    double last_exploration_right_overlap_{0.0};

    // -- LiDAR safety processing ----------------------------------------------

    /**
     * @brief Inflate scan around disparity edges for safety.
     * This means that if a beam has a large jump in range compared
     * to its neighbors, it is probably an edge of an obstacle.
     * Therefore, a safety margin is applied by marking nearby beams as blocked.
     * @param scan Processed scan to modify in place.
     */
    void applyDisparityExtension(ProcessedScan& scan);

    /**
     * @brief Apply uniform wall safety margin to scan ranges.
     * @param scan Processed scan to modify in place.
     */
    void applyWallMargin(ProcessedScan& scan);

    /**
     * @brief Check raw ranges against the rectangular vehicle footprint.
     * @param scan Raw physical ranges after generic preprocessing.
     * @return True if any beam leaves less than the configured body clearance.
     */
    bool violatesFootprintClearance(const ProcessedScan& scan) const;

    // -- Weighted free-space core ---------------------------------------------

    /**
     * @brief Compute per-beam effective clearance profile.
     * @param scan Safety-processed LiDAR scan.
     * @return Effective clearance value for each beam.
     */
    std::vector<double> computeEffectiveClearance(const ProcessedScan& scan);

    /**
     * @brief Compute weighted-centroid steering target.
     * @param scan Safety-processed LiDAR scan.
     * @param eff_clearance Effective clearance profile.
     * @return Target steering angle before smoothing/rate limiting.
     */
    double computeTargetAngle(const ProcessedScan& scan,
                              const std::vector<double>& eff_clearance);

    // Returns [0, 1], where 1 means the projected gap lies on older mapped
    // trajectory and should be treated as a retrace candidate.
    double explorationOverlap(double target_angle,
                              double clearance) const;

    // -- Gap detection (lightweight, for visualisation only) ------------------

    /**
     * @brief Detect contiguous free-space gap segments for visualization.
     * @param scan Safety-processed LiDAR scan.
     * @return Vector of detected gap descriptors.
     */
    std::vector<Gap> findGapsForViz(const ProcessedScan& scan);

    /**
     * @brief Select representative gap from detected candidates.
     * @param gaps Candidate visualization gaps.
     * @return Selected gap descriptor.
     */
    Gap findBestGapForViz(const std::vector<Gap>& gaps);

    // -- Control --------------------------------------------------------------

    /**
     * @brief Compute bounded longitudinal speed command.
     * @param forward_clearance Estimated drivable clearance ahead.
     * @param steering_angle Current steering demand magnitude.
     * @return Commanded longitudinal speed.
     */
    double calculateSpeed(double forward_clearance, double steering_angle);

    /**
     * @brief Apply steering rate limiting.
     * @param target Desired steering angle before rate limiting.
     * @param last Previously emitted steering command.
     * @param dt Control time step.
     * @return Rate-limited steering command.
     */
    double rateLimitSteering(double target, double last, double dt);
};

}  // namespace f1tenth_control

#endif  // F1TENTH_CONTROL_FOLLOW_THE_GAP_HPP_

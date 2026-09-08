#ifndef F1TENTH_CONTROL_FOLLOW_THE_GAP_HPP_
#define F1TENTH_CONTROL_FOLLOW_THE_GAP_HPP_

/**
 * @file follow_the_gap.hpp
 * @brief Contiguous-gap Follow-The-Gap reactive steering controller.
 * @details Selects one connected region of effective free space rather than
 *          averaging disconnected openings. This keeps opposing hairpin
 *          branches from cancelling into a straight command.
 */

#include "common/types.hpp"
#include "common/math_utils.hpp"
#include "common/lidar_processor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace f1tenth_control {

/** @brief Configuration parameters for the mapless FTG controller. */
struct FTGConfig {
    // Vehicle parameters
    double wheelbase{0.324};
    double car_width{0.273};

    // Mapping speed policy
    double max_speed{0.40};
    double min_speed{0.12};
    double speed_full_range{2.0};
    double steer_slowdown_gain{1.0};

    // Steering
    double max_steering{0.5236};
    double steering_gain{1.0};
    double max_steering_rate{3.2};
    double target_ema_alpha{1.0};

    // Effective-clearance gap scoring
    double heading_weight{0.20};
    double score_power{1.5};
    double clearance_cone_scale{1.05};
    double min_score_range{0.25};

    // Small branch hysteresis; this is not a recovery state machine.
    double gap_switch_margin{0.15};
    int gap_switch_scans{2};

    // Phase-2 Ackermann rollout geometry.
    double car_length{0.510};
    double rear_overhang{0.080};
    // Matches the active mapping launch TF: base_link -> lidar x=0.2733 m.
    double lidar_to_rear_axle{0.2733};
    double footprint_margin{0.03};
    double rollout_distance{1.50};
    double rollout_step{0.05};
    double trajectory_min_free_distance{0.25};
    int trajectory_candidate_count{31};

    // Raw physical emergency check
    double emergency_brake_distance{0.10};

    // LiDAR safety and visualization
    double disparity_threshold{0.5};
    double wall_margin{0.03};
    double gap_threshold{0.5};
    double min_gap_width{0.10};

    LidarProcessorConfig lidar_config;
};

/** @brief One connected region of effective, control-usable free space. */
struct DrivableGap {
    size_t start_idx{0};
    size_t end_idx{0};
    double start_angle{0.0};
    double end_angle{0.0};
    double angular_width{0.0};
    double weighted_center_angle{0.0};
    double deepest_angle{0.0};
    double max_clearance{0.0};
    double mean_clearance{0.0};
    double score{0.0};
};

/** @brief Result of selecting a target from one connected gap. */
struct TargetResult {
    bool valid{false};
    double angle{0.0};
};

/** @brief Output and diagnostics for one FTG compute cycle. */
struct FTGOutput {
    DriveCommand command;
    Gap selected_gap;
    size_t closest_point_idx{0};
    double closest_point_dist{0.0};
    bool emergency_stop{false};
    bool no_path{false};
    std::vector<Gap> all_gaps;
    std::vector<DrivableGap> drivable_gaps;
    DrivableGap selected_drivable_gap;
    bool has_selected_drivable_gap{false};

    // Source-timed steering/speed diagnostics.
    double source_stamp_s{0.0};
    double raw_target_angle{0.0};
    double smoothed_target_angle{0.0};
    double raw_steering{0.0};
    double rate_limited_steering{0.0};
    double forward_clearance{0.0};
    double trajectory_free_distance{0.0};
    double trajectory_min_clearance{0.0};
    bool trajectory_collision_free{false};

    ProcessedScan processed_scan;
};

/** @brief Mapless LiDAR-only Follow-The-Gap controller. */
class FollowTheGap {
public:
    explicit FollowTheGap(const FTGConfig& config);

    void setConfig(const FTGConfig& config);

    const FTGConfig& getConfig() const { return config_; }

    /**
     * @brief Compute a command from one LiDAR scan.
     * @param source_stamp_s LiDAR source time in seconds. If omitted, a fixed
     *        25 ms test step is used for backwards-compatible unit calls.
     */
    FTGOutput compute(
        const std::vector<float>& ranges,
        double angle_min,
        double angle_max,
        double angle_increment,
        double source_stamp_s = std::numeric_limits<double>::quiet_NaN()
    );

    const LidarProcessor& getLidarProcessor() const { return lidar_processor_; }

    void reset();

private:
    FTGConfig config_;
    LidarProcessor lidar_processor_;
    double last_steering_{0.0};
    double smoothed_target_{0.0};
    bool first_compute_{true};

    double last_source_stamp_s_{0.0};
    bool has_source_stamp_{false};

    // Minimal branch hysteresis state.
    bool has_selected_gap_{false};
    double previous_gap_angle_{0.0};
    double previous_gap_score_{0.0};
    int pending_alternative_count_{0};

    void applyDisparityExtension(ProcessedScan& scan);
    void applyWallMargin(ProcessedScan& scan);

    std::vector<double> computeEffectiveClearance(const ProcessedScan& scan);

    std::vector<DrivableGap> findDrivableGaps(
        const ProcessedScan& scan,
        const std::vector<double>& eff_clearance
    ) const;

    DrivableGap selectDrivableGap(const std::vector<DrivableGap>& gaps);

    TargetResult computeTargetAngle(const DrivableGap& gap) const;

    double computeDeltaTime(double source_stamp_s);

    bool rawEmergency(const std::vector<float>& ranges,
                      double& closest_range) const;

    struct TrajectoryResult {
        bool valid{false};
        bool collision_free{false};
        double steering{0.0};
        double free_distance{0.0};
        double min_clearance{0.0};
    };

    TrajectoryResult selectTrajectory(
        const std::vector<float>& ranges,
        double angle_min,
        double angle_increment,
        double desired_steering
    ) const;

    std::vector<Gap> findGapsForViz(const ProcessedScan& scan);
    Gap findBestGapForViz(const std::vector<Gap>& gaps);

    double calculateSpeed(double forward_clearance, double steering_angle);
    double rateLimitSteering(double target, double last, double dt);
};

}  // namespace f1tenth_control

#endif  // F1TENTH_CONTROL_FOLLOW_THE_GAP_HPP_

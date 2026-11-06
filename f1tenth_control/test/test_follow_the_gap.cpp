#include <gtest/gtest.h>
#include "algorithms/follow_the_gap.hpp"
#include "common/types.hpp"
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

using namespace f1tenth_control;

class FollowTheGapTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default config for tests (contiguous-gap FTG)
        config_.wheelbase = 0.324;
        config_.car_width = 0.273;
        config_.max_speed = 0.40;
        config_.min_speed = 0.12;
        config_.speed_full_range = 2.0;
        config_.steer_slowdown_gain = 1.0;
        config_.max_steering = 0.5236;
        config_.steering_gain = 1.0;
        config_.max_steering_rate = 100.0;  // Very high for deterministic tests
        config_.target_ema_alpha = 1.0;     // No smoothing for deterministic tests
        config_.emergency_brake_distance = 0.1;

        // Contiguous-gap scoring
        config_.heading_weight = 0.2;
        config_.score_power = 1.5;
        config_.clearance_cone_scale = 1.05;
        config_.min_score_range = 0.25;
        config_.gap_switch_margin = 0.15;
        config_.gap_switch_scans = 2;
        config_.rollout_distance = 0.50;  // Keep synthetic turn fixtures local
        config_.rollout_step = 0.025;
        config_.trajectory_candidate_count = 31;
        // These legacy gap-selection fixtures model abstract rays rather than
        // a physical vehicle. Physical footprint behavior is covered below by
        // dedicated rollout tests using the real AutoDRIVE geometry.
        config_.car_length = 0.0;
        config_.rear_overhang = 0.0;
        config_.car_width = 0.0;
        config_.footprint_margin = 0.0;

        // LiDAR processing
        config_.disparity_threshold = 0.5;
        config_.wall_margin = 0.0;  // Disable for tests
        config_.gap_threshold = 0.8;
        config_.min_gap_width = 0.10;

        // Generic LiDAR preprocessing config
        config_.lidar_config.range_min = 0.1;
        config_.lidar_config.range_max = 12.0;
        config_.lidar_config.angle_min = -constants::PI / 2;
        config_.lidar_config.angle_max = constants::PI / 2;
        config_.lidar_config.apply_median_filter = false;  // Deterministic tests

        ftg_ = std::make_unique<FollowTheGap>(config_);
    }

    // Helper to create a scan with a clear path ahead
    std::vector<float> createOpenScan(size_t num_points) {
        return std::vector<float>(num_points, 8.0f);
    }

    // Helper to create a scan with walls on sides and gap in front
    std::vector<float> createCorridorScan(size_t num_points) {
        std::vector<float> ranges(num_points, 1.0f);  // Walls everywhere
        size_t quarter = num_points / 4;
        for (size_t i = quarter; i < 3 * quarter; ++i) {
            ranges[i] = 6.0f;  // Open in front
        }
        return ranges;
    }

    // Helper to create a scan with obstacle directly ahead
    std::vector<float> createObstacleAheadScan(size_t num_points, float obstacle_dist) {
        std::vector<float> ranges(num_points, 6.0f);
        size_t center = num_points / 2;
        size_t width = num_points / 10;
        for (size_t i = center - width; i <= center + width; ++i) {
            ranges[i] = obstacle_dist;
        }
        return ranges;
    }

    // Standard scan parameters for 180-degree scan with 1-degree resolution
    static constexpr size_t NUM_POINTS = 181;
    static constexpr double ANGLE_MIN = -constants::PI / 2;
    static constexpr double ANGLE_MAX = constants::PI / 2;
    static constexpr double ANGLE_INC = constants::PI / 180.0;

    FTGConfig config_;
    std::unique_ptr<FollowTheGap> ftg_;
};

// =====================================================================
// Basic output tests
// =====================================================================

TEST_F(FollowTheGapTest, ComputeReturnsValidOutput) {
    auto ranges = createOpenScan(NUM_POINTS);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_GT(output.command.speed, 0.0);
    EXPECT_FALSE(output.emergency_stop);
    EXPECT_FALSE(output.processed_scan.filtered_ranges.empty());
}

TEST_F(FollowTheGapTest, OpenPathDrivesStraight) {
    auto ranges = createOpenScan(NUM_POINTS);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    // In a symmetric open path the weighted centroid should be near zero
    EXPECT_NEAR(output.command.steering_angle, 0.0, 0.05);
    EXPECT_GT(output.command.speed, 0.0);
}

TEST_F(FollowTheGapTest, CorridorDrivesStraight) {
    auto ranges = createCorridorScan(NUM_POINTS);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    // Corridor: open in front, walls on sides -> should steer roughly straight
    EXPECT_NEAR(output.command.steering_angle, 0.0, 0.15);
    EXPECT_GT(output.command.speed, 0.0);
}

// =====================================================================
// Emergency stop tests
// =====================================================================

TEST_F(FollowTheGapTest, EmergencyStopWhenObstacleTooClose) {
    // All obstacles very close
    std::vector<float> ranges(NUM_POINTS, 0.05f);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_TRUE(output.emergency_stop);
    EXPECT_DOUBLE_EQ(output.command.speed, 0.0);
}

TEST_F(FollowTheGapTest, EmergencyStopDistanceConfigurable) {
    config_.emergency_brake_distance = 0.5;
    ftg_->setConfig(config_);

    std::vector<float> ranges(NUM_POINTS, 0.4f);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_TRUE(output.emergency_stop);
}

TEST_F(FollowTheGapTest, NoEmergencyStopWhenFarEnough) {
    config_.emergency_brake_distance = 0.1;
    ftg_->setConfig(config_);

    auto ranges = createOpenScan(NUM_POINTS);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_FALSE(output.emergency_stop);
}

// =====================================================================
// Gap detection for visualisation
// =====================================================================

TEST_F(FollowTheGapTest, DetectsGaps) {
    auto ranges = createCorridorScan(NUM_POINTS);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    // Should find at least one gap in the corridor
    EXPECT_FALSE(output.all_gaps.empty());
}

// =====================================================================
// Steering direction tests
// =====================================================================

TEST_F(FollowTheGapTest, SteersTowardOpenSpace) {
    // Gap only on the right (indices 0..quarter), walls elsewhere
    std::vector<float> ranges(NUM_POINTS, 0.5f);  // close walls
    size_t quarter = NUM_POINTS / 4;
    for (size_t i = 0; i < quarter; ++i) {
        ranges[i] = 6.0f;  // Open on the right (negative angles)
    }
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    // Steering angle should be negative (right)
    EXPECT_LT(output.command.steering_angle, -0.01);
}

TEST_F(FollowTheGapTest, SteersLeftWhenGapOnLeft) {
    std::vector<float> ranges(NUM_POINTS, 0.5f);
    size_t three_quarter = 3 * NUM_POINTS / 4;
    for (size_t i = three_quarter; i < NUM_POINTS; ++i) {
        ranges[i] = 6.0f;  // Open on the left (positive angles)
    }
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_GT(output.command.steering_angle, 0.01);
}

TEST_F(FollowTheGapTest, SteeringAngleClamped) {
    // Very extreme gap scenario — steering must stay within max_steering
    std::vector<float> ranges(NUM_POINTS, 0.5f);
    // Only last 10 beams are open
    for (size_t i = NUM_POINTS - 10; i < NUM_POINTS; ++i) {
        ranges[i] = 8.0f;
    }
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_LE(std::abs(output.command.steering_angle), config_.max_steering + 0.001);
}

// =====================================================================
// Speed tests
// =====================================================================

TEST_F(FollowTheGapTest, SpeedWithinLimits) {
    auto ranges = createOpenScan(NUM_POINTS);
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_GE(output.command.speed, config_.min_speed);
    EXPECT_LE(output.command.speed, config_.max_speed);
}

TEST_F(FollowTheGapTest, SpeedReducesWhenTurning) {
    // Open path -> near max speed with small steering
    auto open_ranges = createOpenScan(NUM_POINTS);
    auto open_output = ftg_->compute(open_ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    // Gap on one side -> turns -> speed should be <= open-path speed
    std::vector<float> ranges(NUM_POINTS, 0.5f);
    for (size_t i = 0; i < NUM_POINTS / 4; ++i) {
        ranges[i] = 6.0f;
    }
    ftg_->reset();
    auto turn_output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_LE(turn_output.command.speed, open_output.command.speed + 0.01);
}

// =====================================================================
// Configuration tests
// =====================================================================

TEST_F(FollowTheGapTest, ConfigurationCanBeUpdated) {
    auto ranges = createOpenScan(NUM_POINTS);
    auto output1 = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    config_.max_speed = 1.5;
    ftg_->setConfig(config_);
    ftg_->reset();
    auto output2 = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_LE(output2.command.speed, 1.5);
}

// =====================================================================
// Empty / degenerate scan tests
// =====================================================================

TEST_F(FollowTheGapTest, HandlesEmptyScan) {
    std::vector<float> empty_ranges;
    auto output = ftg_->compute(empty_ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_TRUE(output.emergency_stop);
    EXPECT_DOUBLE_EQ(output.command.speed, 0.0);
}

TEST_F(FollowTheGapTest, HandlesSinglePoint) {
    std::vector<float> ranges = {5.0f};
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    // Single point: should not crash
    EXPECT_FALSE(output.processed_scan.filtered_ranges.empty());
}

// =====================================================================
// Weighted free-space specific tests
// =====================================================================

TEST_F(FollowTheGapTest, SymmetricScanProducesCenteredSteering) {
    // Perfectly symmetric corridor -> target should be near 0
    std::vector<float> ranges(NUM_POINTS);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        double angle = ANGLE_MIN + i * ANGLE_INC;
        // Symmetric U-shape: close on sides, far in front
        ranges[i] = static_cast<float>(2.0 + 4.0 * std::cos(angle) * std::cos(angle));
    }
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_NEAR(output.command.steering_angle, 0.0, 0.05);
}

TEST_F(FollowTheGapTest, HeadingWeightDoesNotSuppressStrongSideGap) {
    // Block the straight direction, provide a weak narrow forward opening and
    // a much deeper side opening. A strong side gap must remain selectable.
    std::vector<float> ranges(NUM_POINTS, 0.15f);
    for (size_t i = 80; i <= 100; ++i) {
        ranges[i] = 0.8f;  // weak forward opening
    }
    for (size_t i = 145; i < NUM_POINTS; ++i) {
        ranges[i] = 6.0f;  // strong left opening
    }

    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    ASSERT_TRUE(output.has_selected_drivable_gap);
    EXPECT_GT(output.selected_drivable_gap.weighted_center_angle, 0.6);
    EXPECT_GT(output.command.steering_angle, 0.01);
}

TEST_F(FollowTheGapTest, DisconnectedOpposingGapsDoNotAverageToStraight) {
    std::vector<float> ranges(NUM_POINTS, 0.15f);
    for (size_t i = 0; i <= 35; ++i) {
        ranges[i] = 6.0f;  // right opening
    }
    for (size_t i = 145; i < NUM_POINTS; ++i) {
        ranges[i] = 6.0f;  // opposing left opening
    }

    const auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    EXPECT_GE(output.drivable_gaps.size(), 2u);
    ASSERT_TRUE(output.has_selected_drivable_gap);
    EXPECT_GT(std::abs(output.raw_target_angle), 0.5);
    EXPECT_GT(std::abs(output.command.steering_angle), 0.01);
}

TEST_F(FollowTheGapTest, SelectedTargetRemainsInsideSelectedGap) {
    const auto output = ftg_->compute(
        createCorridorScan(NUM_POINTS), ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    ASSERT_TRUE(output.has_selected_drivable_gap);
    const auto& gap = output.selected_drivable_gap;
    EXPECT_GE(output.raw_target_angle, gap.start_angle - 1e-9);
    EXPECT_LE(output.raw_target_angle, gap.end_angle + 1e-9);
}

TEST_F(FollowTheGapTest, NoValidGapStopsWithoutStraightFallback) {
    const std::vector<float> ranges(NUM_POINTS, 0.20f);

    const auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    EXPECT_TRUE(output.no_path);
    EXPECT_FALSE(output.emergency_stop);
    EXPECT_DOUBLE_EQ(output.command.speed, 0.0);
}

TEST_F(FollowTheGapTest, SourceTimestampsControlSteeringRate) {
    config_.max_steering_rate = 1.0;
    auto first_controller = std::make_unique<FollowTheGap>(config_);
    auto delayed_controller = std::make_unique<FollowTheGap>(config_);
    std::vector<float> ranges(NUM_POINTS, 0.15f);
    for (size_t i = 145; i < NUM_POINTS; ++i) {
        ranges[i] = 6.0f;
    }

    first_controller->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC, 10.0);
    const auto first_second = first_controller->compute(
        ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC, 10.1);

    delayed_controller->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC, 20.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    const auto delayed_second = delayed_controller->compute(
        ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC, 20.1);

    EXPECT_NEAR(first_second.command.steering_angle,
                delayed_second.command.steering_angle, 1e-12);
}

TEST_F(FollowTheGapTest, DemandedSteeringReducesSpeedBeforeRateLimitCatchesUp) {
    config_.max_steering_rate = 0.1;
    auto controller = std::make_unique<FollowTheGap>(config_);
    const auto open_output = controller->compute(
        createOpenScan(NUM_POINTS), ANGLE_MIN, ANGLE_MAX, ANGLE_INC, 1.0);

    std::vector<float> turn_scan(NUM_POINTS, 0.20f);
    for (size_t i = 80; i <= 100; ++i) {
        turn_scan[i] = 0.35f;  // retain a small forward clearance estimate
    }
    for (size_t i = 145; i < NUM_POINTS; ++i) {
        turn_scan[i] = 6.0f;   // sharp left branch
    }
    controller->reset();
    const auto turn_output = controller->compute(
        turn_scan, ANGLE_MIN, ANGLE_MAX, ANGLE_INC, 2.0);

    EXPECT_GT(std::abs(turn_output.raw_steering), 0.45);
    EXPECT_LT(std::abs(turn_output.rate_limited_steering), 0.02);
    EXPECT_LT(turn_output.command.speed, open_output.command.speed - 0.05);
}

TEST_F(FollowTheGapTest, RawCloseReturnTriggersEmergencyBelowNavigationMinimum) {
    std::vector<float> ranges(NUM_POINTS, 6.0f);
    ranges[NUM_POINTS / 2] = 0.05f;  // below lidar.range_min, but physically real

    const auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    EXPECT_TRUE(output.emergency_stop);
    EXPECT_DOUBLE_EQ(output.command.speed, 0.0);
}

TEST_F(FollowTheGapTest, HairpinBranchDoesNotFlickerAcrossStableScans) {
    std::vector<float> right_gap(NUM_POINTS, 0.15f);
    for (size_t i = 0; i <= 35; ++i) {
        right_gap[i] = 6.0f;
    }

    for (int scan = 0; scan < 4; ++scan) {
        const auto output = ftg_->compute(
            right_gap, ANGLE_MIN, ANGLE_MAX, ANGLE_INC, 30.0 + 0.05 * scan);
        ASSERT_TRUE(output.has_selected_drivable_gap);
        EXPECT_LT(output.selected_drivable_gap.weighted_center_angle, -0.5);
        EXPECT_LT(output.command.steering_angle, -0.01);
    }
}

TEST_F(FollowTheGapTest, SweptFootprintRejectsRayClearForwardObstacle) {
    config_.car_length = 0.510;
    config_.rear_overhang = 0.080;
    config_.car_width = 0.273;
    config_.lidar_to_rear_axle = 0.2733;
    config_.footprint_margin = 0.03;
    config_.rollout_distance = 1.0;
    config_.rollout_step = 0.02;
    ftg_->setConfig(config_);
    ftg_->reset();

    auto ranges = createOpenScan(NUM_POINTS);
    // This return is outside the current footprint and therefore not an
    // emergency, but a straight rollout would reach it after about 0.25 m.
    ranges[NUM_POINTS / 2] = 0.45f;

    const auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);

    EXPECT_FALSE(output.emergency_stop);
    EXPECT_TRUE(output.trajectory_collision_free);
    EXPECT_GT(std::abs(output.command.steering_angle), 0.01);
}

TEST_F(FollowTheGapTest, FullScanRearSectorIsIncludedInFootprintCheck) {
    config_.car_length = 0.510;
    config_.rear_overhang = 0.080;
    config_.car_width = 0.273;
    config_.lidar_to_rear_axle = 0.2733;
    config_.footprint_margin = 0.03;
    config_.rollout_distance = 1.0;
    config_.rollout_step = 0.02;
    ftg_->setConfig(config_);
    ftg_->reset();

    constexpr size_t FULL_POINTS = 361;
    constexpr double FULL_MIN = -constants::PI;
    constexpr double FULL_INC = constants::PI / 180.0;
    auto ranges = std::vector<float>(FULL_POINTS, 8.0f);
    // 150 degrees / 0.30 m is outside the navigation sector but intersects
    // the expanded rear-side footprint at the current pose.
    ranges[330] = 0.30f;

    const auto output = ftg_->compute(
        ranges, FULL_MIN, constants::PI, FULL_INC);

    EXPECT_FALSE(output.emergency_stop);
    EXPECT_TRUE(output.no_path);
    EXPECT_FALSE(output.trajectory_collision_free);
    EXPECT_DOUBLE_EQ(output.command.speed, 0.0);
}

TEST_F(FollowTheGapTest, ResetClearsState) {
    auto ranges = createOpenScan(NUM_POINTS);
    ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    ftg_->reset();
    // After reset, first_compute should be true again -> no crash
    auto output = ftg_->compute(ranges, ANGLE_MIN, ANGLE_MAX, ANGLE_INC);
    EXPECT_FALSE(output.emergency_stop);
}

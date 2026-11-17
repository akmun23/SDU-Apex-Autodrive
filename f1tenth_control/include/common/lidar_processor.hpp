#ifndef F1TENTH_CONTROL_LIDAR_PROCESSOR_HPP_
#define F1TENTH_CONTROL_LIDAR_PROCESSOR_HPP_

/** Generic scan filtering; disparity extension and safety bubbles belong to the
 * calling algorithm. Concurrent reads are safe after construction;
 * `setConfig()` requires exclusive access. */

#include "common/types.hpp"
#include "common/math_utils.hpp"
#include <vector>
#include <cmath>
#include <limits>

namespace f1tenth_control {


struct LidarProcessorConfig {
    // Range filtering
    double range_min{0.1};           // Minimum valid range (m)
    double range_max{10.0};          // Maximum valid range (m)
    
    // Angular filtering (process only this range)
    double angle_min{-2.35};         // Minimum angle to process (rad)
    double angle_max{2.35};          // Maximum angle to process (rad)
    
    // Noise filtering
    bool apply_median_filter{true};  // Apply median filter
    size_t median_window_size{3};    // Median filter window size
};


class LidarProcessor {
public:
    explicit LidarProcessor(const LidarProcessorConfig& config = LidarProcessorConfig());
    
    void setConfig(const LidarProcessorConfig& config);

    const LidarProcessorConfig& getConfig() const { return config_; }
    
    /** Filter ranges in metres and beam angles in radians; return clipped ranges,
     * a validity mask and beam angles. */
    ProcessedScan processScan(
        const std::vector<float>& ranges,
        double angle_min,
        double angle_max,
        double angle_increment
    );
    
    size_t findClosestPoint(const ProcessedScan& scan);

    /** Cartesian point in the robot frame. */
    Point2D scanPointToCartesian(const ProcessedScan& scan, size_t index);
    
    /** Map-frame boundary points with the supplied timestamp. */
    std::vector<BoundaryPoint> extractBoundaryPoints(
        const ProcessedScan& scan,
        const Pose2D& robot_pose,
        double timestamp
    );

private:
    LidarProcessorConfig config_;
    
    void applyMedianFilter(std::vector<double>& ranges);
    
    void validateRanges(ProcessedScan& scan);
};

}  // namespace f1tenth_control

#endif  // F1TENTH_CONTROL_LIDAR_PROCESSOR_HPP_

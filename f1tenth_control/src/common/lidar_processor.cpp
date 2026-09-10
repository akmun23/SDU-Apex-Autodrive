#include "common/lidar_processor.hpp"

namespace f1tenth_control {

LidarProcessor::LidarProcessor(const LidarProcessorConfig& config)
    : config_(config) {}

void LidarProcessor::setConfig(const LidarProcessorConfig& config) {
    config_ = config;
}

ProcessedScan LidarProcessor::processScan(
    const std::vector<float>& ranges,
    double angle_min,
    double angle_max,
    double angle_increment
) {
    ProcessedScan scan;
    scan.angle_min = angle_min;
    scan.angle_max = angle_max;
    scan.angle_increment = angle_increment;
    scan.range_min = config_.range_min;
    scan.range_max = config_.range_max;

    const size_t n = ranges.size();
    scan.filtered_ranges.resize(n);
    scan.angles.resize(n);
    scan.valid.resize(n);

    for (size_t i = 0; i < n; ++i) {
        const double angle = angle_min + i * angle_increment;
        const double range = static_cast<double>(ranges[i]);
        const bool in_range = angle >= config_.angle_min
            && angle <= config_.angle_max;
        const bool valid_range = std::isfinite(range)
            && range >= config_.range_min;

        scan.filtered_ranges[i] = range;
        scan.angles[i] = angle;
        scan.valid[i] = in_range && valid_range;
    }

    if (config_.apply_median_filter && config_.median_window_size > 1) {
        applyMedianFilter(scan.filtered_ranges);
    }
    validateRanges(scan);
    return scan;
}

void LidarProcessor::applyMedianFilter(std::vector<double>& ranges) {
    ranges = math::medianFilter(ranges, config_.median_window_size);
}

void LidarProcessor::validateRanges(ProcessedScan& scan) {
    for (size_t i = 0; i < scan.filtered_ranges.size(); ++i) {
        double& range = scan.filtered_ranges[i];
        if (!std::isfinite(range) || range < config_.range_min) {
            range = config_.range_min;
            scan.valid[i] = false;
        } else if (range > config_.range_max) {
            range = config_.range_max;
        }
    }
}

size_t LidarProcessor::findClosestPoint(const ProcessedScan& scan) {
    if (scan.filtered_ranges.empty()) return 0;

    size_t closest_idx = 0;
    double min_range = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < scan.filtered_ranges.size(); ++i) {
        const double angle = scan.angles[i];
        if (angle < config_.angle_min || angle > config_.angle_max) {
            continue;
        }
        if (scan.valid[i] && scan.filtered_ranges[i] < min_range) {
            min_range = scan.filtered_ranges[i];
            closest_idx = i;
        }
    }
    return closest_idx;
}

}  // namespace f1tenth_control

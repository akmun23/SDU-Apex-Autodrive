#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace gpu_amcl_cpp::scan_likelihood {

// These bounds and the score definition match the offline held-out experiment
// in tools/model_id/analyze_amcl_scan_observability.py.
constexpr int kSearchHalfSteps = 20;
constexpr double kSearchStepM = 0.01;
constexpr double kScoreTieTolerance = 1.0e-7;
constexpr double kPi = 3.14159265358979323846;

struct LikelihoodFieldConfig {
    int max_beams = 0;
    double z_hit = 0.0;
    double z_rand = 0.0;
    double sigma_hit = 0.0;
    double laser_min_range = 0.0;
    double laser_max_range = 0.0;
    double laser_offset_x = 0.0;
    double laser_offset_y = 0.0;
    bool normalize_likelihood_by_beams = true;
    double likelihood_scale = 1.0;
};

struct AlongTrackOffset {
    bool valid = false;
    double offset_m = 0.0;
    double score_gain = 0.0;
    std::size_t valid_beams = 0;
    std::size_t sampled_beams = 0;
    bool at_search_boundary = false;
};

/**
 * Score a source LaserScan against the AMCL distance field at 41 poses
 * displaced along the causal pose heading by [-0.20, +0.20] m. The map
 * processor currently represents axis-aligned OccupancyGrid maps; this is
 * therefore identical to the production model for the configured track map
 * (origin yaw 0).
 */
inline AlongTrackOffset estimate_along_track_offset(
        const std::vector<float>& ranges,
        double angle_min,
        double angle_increment,
        double pose_x,
        double pose_y,
        double pose_yaw,
        int map_width,
        int map_height,
        double map_resolution,
        double map_origin_x,
        double map_origin_y,
        const std::vector<float>& distance_field,
        const LikelihoodFieldConfig& config) {
    AlongTrackOffset result;
    if (ranges.empty() || map_width <= 0 || map_height <= 0 ||
        !(map_resolution > 0.0) ||
        distance_field.size() != static_cast<std::size_t>(map_width) *
                                     static_cast<std::size_t>(map_height) ||
        config.max_beams <= 0 || !(config.sigma_hit > 0.0) ||
        !(config.laser_max_range > 0.0) ||
        !std::isfinite(config.z_hit) || !std::isfinite(config.z_rand) ||
        !std::isfinite(config.sigma_hit) ||
        !std::isfinite(config.laser_min_range) ||
        !std::isfinite(config.laser_max_range) ||
        !std::isfinite(config.likelihood_scale) ||
        !std::isfinite(pose_x) || !std::isfinite(pose_y) ||
        !std::isfinite(pose_yaw) || !std::isfinite(angle_min) ||
        !std::isfinite(angle_increment) || !std::isfinite(map_resolution) ||
        !std::isfinite(map_origin_x) || !std::isfinite(map_origin_y)) {
        return result;
    }

    const std::size_t beam_step = std::max<std::size_t>(
        1, ranges.size() / static_cast<std::size_t>(config.max_beams));
    std::array<double, 2 * kSearchHalfSteps + 1> scores{};
    scores.fill(0.0);
    const double cos_yaw = std::cos(pose_yaw);
    const double sin_yaw = std::sin(pose_yaw);
    const double normalizer = 1.0 /
        (std::sqrt(2.0 * kPi) * config.sigma_hit);
    const double inv_two_sigma_squared =
        -0.5 / (config.sigma_hit * config.sigma_hit);
    std::size_t valid_beams = 0;

    for (std::size_t beam = 0; beam < ranges.size(); beam += beam_step) {
        const double range = static_cast<double>(ranges[beam]);
        if (!std::isfinite(range) || range < config.laser_min_range ||
            range > config.laser_max_range) {
            continue;
        }
        ++valid_beams;
        const double beam_angle = angle_min +
            static_cast<double>(beam) * angle_increment;
        const double world_angle = pose_yaw + beam_angle;
        const double endpoint_dx = range * std::cos(world_angle);
        const double endpoint_dy = range * std::sin(world_angle);

        for (int step = -kSearchHalfSteps; step <= kSearchHalfSteps; ++step) {
            const std::size_t score_index = static_cast<std::size_t>(
                step + kSearchHalfSteps);
            const double candidate_x = pose_x +
                static_cast<double>(step) * kSearchStepM * cos_yaw;
            const double candidate_y = pose_y +
                static_cast<double>(step) * kSearchStepM * sin_yaw;
            const double candidate_laser_x = candidate_x +
                config.laser_offset_x * cos_yaw -
                config.laser_offset_y * sin_yaw;
            const double candidate_laser_y = candidate_y +
                config.laser_offset_x * sin_yaw +
                config.laser_offset_y * cos_yaw;
            const double endpoint_x = candidate_laser_x + endpoint_dx;
            const double endpoint_y = candidate_laser_y + endpoint_dy;
            const double fx = (endpoint_x - map_origin_x) /
                map_resolution - 0.5;
            const double fy = (endpoint_y - map_origin_y) /
                map_resolution - 0.5;

            double distance = config.laser_max_range;
            if (fx >= -0.5 && fx < static_cast<double>(map_width) - 0.5 &&
                fy >= -0.5 && fy < static_cast<double>(map_height) - 0.5) {
                const int x0_raw = static_cast<int>(std::floor(fx));
                const int y0_raw = static_cast<int>(std::floor(fy));
                const double sx = fx - static_cast<double>(x0_raw);
                const double sy = fy - static_cast<double>(y0_raw);
                const int x0 = std::clamp(x0_raw, 0, map_width - 1);
                const int x1 = std::clamp(x0_raw + 1, 0, map_width - 1);
                const int y0 = std::clamp(y0_raw, 0, map_height - 1);
                const int y1 = std::clamp(y0_raw + 1, 0, map_height - 1);
                const auto at = [&](int x, int y) {
                    return static_cast<double>(distance_field[
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(map_width) +
                        static_cast<std::size_t>(x)]);
                };
                const double d00 = at(x0, y0);
                const double d10 = at(x1, y0);
                const double d01 = at(x0, y1);
                const double d11 = at(x1, y1);
                const double d0 = d00 + sx * (d10 - d00);
                const double d1 = d01 + sx * (d11 - d01);
                distance = d0 + sy * (d1 - d0);
            }

            const double probability = config.z_hit * normalizer *
                std::exp(inv_two_sigma_squared * distance * distance) +
                config.z_rand / config.laser_max_range;
            if (!std::isfinite(probability) || probability <= 0.0) {
                return result;
            }
            scores[score_index] += std::log(std::max(probability, 1.0e-30));
        }
    }

    result.valid_beams = valid_beams;
    result.sampled_beams = (ranges.size() + beam_step - 1) / beam_step;
    if (valid_beams == 0) {
        return result;
    }
    if (config.normalize_likelihood_by_beams) {
        for (double& score : scores) {
            score /= static_cast<double>(valid_beams);
        }
    }
    const double score_scale = std::max(config.likelihood_scale, 0.0);
    for (double& score : scores) {
        score *= score_scale;
    }

    const std::size_t center_index = static_cast<std::size_t>(kSearchHalfSteps);
    const double best_score = *std::max_element(scores.begin(), scores.end());
    if (!std::isfinite(best_score) || !std::isfinite(scores[center_index])) {
        return result;
    }
    int best_step = -kSearchHalfSteps;
    for (int step = -kSearchHalfSteps; step <= kSearchHalfSteps; ++step) {
        const std::size_t index = static_cast<std::size_t>(
            step + kSearchHalfSteps);
        if (scores[index] >= best_score - kScoreTieTolerance &&
            std::abs(step) < std::abs(best_step)) {
            best_step = step;
        }
    }

    result.valid = true;
    result.offset_m = static_cast<double>(best_step) * kSearchStepM;
    result.score_gain = best_score - scores[center_index];
    if (!std::isfinite(result.offset_m) || !std::isfinite(result.score_gain)) {
        return AlongTrackOffset{};
    }
    result.at_search_boundary =
        std::abs(best_step) == kSearchHalfSteps;
    return result;
}

}  // namespace gpu_amcl_cpp::scan_likelihood

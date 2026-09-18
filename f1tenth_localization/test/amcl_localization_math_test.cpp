#include "gpu_amcl_cpp/helpers/localization_math.hpp"
#include "gpu_amcl_cpp/helpers/scan_likelihood_along_track.hpp"
#include "gpu_amcl_cpp/helpers/scan_validity.hpp"

#include <Eigen/Eigenvalues>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  using gpu_amcl_cpp::localization_math::PoseInterpolation;
  using gpu_amcl_cpp::localization_math::PoseSample;

  const auto empty_scan = gpu_amcl_cpp::scan_validity::count_sampled_valid_ranges(
      std::vector<float>(1081, std::numeric_limits<float>::infinity()),
      270, 0.06, 10.0);
  assert(empty_scan.sampled == 271);
  assert(empty_scan.valid == 0);

  const auto sparse_scan = gpu_amcl_cpp::scan_validity::count_sampled_valid_ranges(
      {1.0f, 0.0f, 0.05f, 2.0f, std::numeric_limits<float>::quiet_NaN(),
       3.0f, std::numeric_limits<float>::infinity(), 10.0f},
      3, 0.06, 10.0);
  // The production kernel uses step=floor(8/3)=2, so indices 0,2,4,6
  // are sampled; index 0 is the only valid return in that subset.
  assert(sparse_scan.sampled == 4);
  assert(sparse_scan.valid == 1);

  // A synthetic vertical wall at x=2.525 m has a 2.475 m return from a
  // vehicle pose at x=0.05 m. The bounded scan-likelihood search must recover
  // that +5 cm causal along-track correction, matching the offline fitter.
  constexpr int map_width = 200;
  constexpr int map_height = 200;
  constexpr double map_resolution = 0.05;
  std::vector<float> distance_field(
      static_cast<std::size_t>(map_width * map_height));
  for (int y = 0; y < map_height; ++y) {
    for (int x = 0; x < map_width; ++x) {
      distance_field[static_cast<std::size_t>(y * map_width + x)] =
          static_cast<float>(std::abs(x - 150) * map_resolution);
    }
  }
  gpu_amcl_cpp::scan_likelihood::LikelihoodFieldConfig score_cfg;
  score_cfg.max_beams = 270;
  score_cfg.z_hit = 0.90;
  score_cfg.z_rand = 0.10;
  score_cfg.sigma_hit = 0.05;
  score_cfg.laser_min_range = 0.06;
  score_cfg.laser_max_range = 10.0;
  score_cfg.laser_offset_x = 0.0;
  score_cfg.laser_offset_y = 0.0;
  score_cfg.normalize_likelihood_by_beams = true;
  score_cfg.likelihood_scale = 2.0;
  const auto score_correction =
      gpu_amcl_cpp::scan_likelihood::estimate_along_track_offset(
          {2.475f}, 0.0, 0.0, 0.0, 0.0, 0.0,
          map_width, map_height, map_resolution, -5.0, -5.0,
          distance_field, score_cfg);
  if (!score_correction.valid || score_correction.valid_beams != 1 ||
      std::abs(score_correction.offset_m - 0.05) >= 1.0e-12 ||
      !(score_correction.score_gain > 0.0) ||
      !std::isfinite(score_correction.score_gain) ||
      score_correction.at_search_boundary) {
    std::cerr << "scan likelihood did not recover the finite +5 cm test offset"
              << std::endl;
    return 1;
  }
  const auto invalid_score =
      gpu_amcl_cpp::scan_likelihood::estimate_along_track_offset(
          {std::numeric_limits<float>::infinity()}, 0.0, 0.0,
          0.0, 0.0, 0.0, map_width, map_height, map_resolution,
          -5.0, -5.0, distance_field, score_cfg);
  if (invalid_score.valid || invalid_score.valid_beams != 0) {
    std::cerr << "scan likelihood accepted a scan with no finite returns"
              << std::endl;
    return 1;
  }

  const std::vector<PoseSample> samples{{1.0, 0.0, 0.0, 3.10},
                                        {1.1, 1.0, 2.0, -3.10}};
  PoseInterpolation match;
  assert(!gpu_amcl_cpp::localization_math::interpolate_pose(samples, 0.9, match));
  assert(!gpu_amcl_cpp::localization_math::interpolate_pose(samples, 1.2, match));
  assert(gpu_amcl_cpp::localization_math::interpolate_pose(samples, 1.05, match));
  assert(std::abs(match.pose.x - 0.5) < 1.0e-12);
  assert(std::abs(match.pose.y - 1.0) < 1.0e-12);
  assert(std::abs(match.pose.theta - 3.141592653589793) < 0.05);
  assert(match.bracket_before == 1.0);
  assert(match.bracket_after == 1.1);

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  covariance(0, 1) = 0.01;
  covariance(1, 0) = 0.01;
  const auto one_second =
      gpu_amcl_cpp::localization_math::grow_pose_covariance(covariance, 1.0, 0.002, 0.0005);
  auto twenty_hz = covariance;
  for (int i = 0; i < 20; ++i) {
    twenty_hz = gpu_amcl_cpp::localization_math::grow_pose_covariance(
        twenty_hz, 0.05, 0.002, 0.0005);
  }
  assert(std::abs(one_second(0, 0) - 0.002) < 1.0e-12);
  assert(std::abs(one_second(1, 1) - 0.002) < 1.0e-12);
  assert(std::abs(one_second(2, 2) - 0.0005) < 1.0e-12);
  assert((one_second - twenty_hz).norm() < 1.0e-12);
  const auto no_backward_growth =
      gpu_amcl_cpp::localization_math::grow_pose_covariance(one_second, -0.5, 1.0, 1.0);
  assert((no_backward_growth - one_second).norm() < 1.0e-12);

  // The EKF and AMCL use the same SE(2) covariance convention. A heading
  // uncertainty must project into the lateral position uncertainty when the
  // vehicle moves, and the result must remain symmetric positive semidefinite.
  Eigen::Matrix3d pose_covariance = Eigen::Matrix3d::Zero();
  pose_covariance(2, 2) = 0.04;
  Eigen::Matrix3d delta_covariance = Eigen::Matrix3d::Zero();
  delta_covariance(0, 0) = 0.002;
  delta_covariance(1, 1) = 0.003;
  delta_covariance(2, 2) = 0.001;
  const Eigen::Vector3d pose(0.0, 0.0, 1.5707963267948966);
  const Eigen::Vector3d delta(1.0, 0.0, 0.1);
  const auto propagated =
      gpu_amcl_cpp::localization_math::propagate_pose_covariance(
          pose_covariance, pose, delta, delta_covariance);
  assert(std::abs(propagated(0, 0) - 0.043) < 1.0e-12);
  assert(std::abs(propagated(1, 1) - 0.002) < 1.0e-12);
  assert((propagated - propagated.transpose()).norm() < 1.0e-12);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(propagated);
  assert(eig.eigenvalues().minCoeff() >= -1.0e-12);

  std::cout << "amcl_localization_math_test: PASS" << std::endl;
  return 0;
}

#include "gpu_amcl_cpp/helpers/localization_math.hpp"

#include <Eigen/Eigenvalues>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  using gpu_amcl_cpp::localization_math::PoseInterpolation;
  using gpu_amcl_cpp::localization_math::PoseSample;

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

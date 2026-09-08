#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpu_amcl_cpp::localization_math {

struct PoseSample {
    double stamp = 0.0;
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
};

struct PoseInterpolation {
    PoseSample pose;
    double bracket_before = 0.0;
    double bracket_after = 0.0;
};

inline double angle_diff(double a, double b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

// Interpolate only inside the source-time interval. A future query must wait
// for a newer source sample; clamping it to the latest sample creates a
// systematic one-packet-old localization state.
inline bool interpolate_pose(const std::vector<PoseSample>& samples,
                             double stamp,
                             PoseInterpolation& result) {
    if (samples.empty() || stamp < samples.front().stamp ||
        stamp > samples.back().stamp) {
        return false;
    }

    if (stamp == samples.front().stamp) {
        result.pose = samples.front();
        result.bracket_before = stamp;
        result.bracket_after = stamp;
        return true;
    }
    if (stamp == samples.back().stamp) {
        result.pose = samples.back();
        result.bracket_before = stamp;
        result.bracket_after = stamp;
        return true;
    }

    for (size_t i = 1; i < samples.size(); ++i) {
        const auto& before = samples[i - 1];
        const auto& after = samples[i];
        if (stamp > after.stamp) {
            continue;
        }
        result.bracket_before = before.stamp;
        result.bracket_after = after.stamp;
        const double dt = after.stamp - before.stamp;
        if (dt <= 0.0) {
            result.pose = after;
            result.pose.stamp = stamp;
            return true;
        }
        const double ratio = (stamp - before.stamp) / dt;
        result.pose.stamp = stamp;
        result.pose.x = before.x + ratio * (after.x - before.x);
        result.pose.y = before.y + ratio * (after.y - before.y);
        result.pose.theta = before.theta + ratio * angle_diff(after.theta, before.theta);
        return true;
    }
    return false;
}

inline Eigen::Matrix3d grow_pose_covariance(
    const Eigen::Matrix3d& covariance,
    double dt_s,
    double process_xy_m2_per_s,
    double process_yaw2_per_s) {
    Eigen::Matrix3d result = covariance;
    if (!result.allFinite()) {
        result.setZero();
    }
    const double dt = std::max(0.0, dt_s);
    result(0, 0) += std::max(0.0, process_xy_m2_per_s) * dt;
    result(1, 1) += std::max(0.0, process_xy_m2_per_s) * dt;
    result(2, 2) += std::max(0.0, process_yaw2_per_s) * dt;
    result = 0.5 * (result + result.transpose());
    for (int i = 0; i < 3; ++i) {
        result(i, i) = std::max(0.0, result(i, i));
    }
    return result;
}

// Propagate a planar pose covariance through
//     pose_next = pose \oplus delta,
// where delta is expressed in the current pose frame.  This is the same
// first-order SE(2) model used by the odometry trust filter and AMCL's
// current-map pose propagation.  Keeping it in one header makes the runtime
// and deterministic tests exercise the same covariance convention.
inline Eigen::Matrix3d propagate_pose_covariance(
    const Eigen::Matrix3d& covariance,
    const Eigen::Vector3d& pose,
    const Eigen::Vector3d& delta,
    const Eigen::Matrix3d& delta_covariance) {
    Eigen::Matrix3d state_covariance = covariance;
    Eigen::Matrix3d motion_covariance = delta_covariance;
    if (!state_covariance.allFinite()) {
        state_covariance.setZero();
    }
    if (!motion_covariance.allFinite()) {
        motion_covariance.setZero();
    }

    const double c = std::cos(pose[2]);
    const double s = std::sin(pose[2]);
    Eigen::Matrix3d state_jacobian = Eigen::Matrix3d::Identity();
    state_jacobian(0, 2) = -s * delta[0] - c * delta[1];
    state_jacobian(1, 2) =  c * delta[0] - s * delta[1];

    Eigen::Matrix3d motion_jacobian = Eigen::Matrix3d::Identity();
    motion_jacobian(0, 0) = c;
    motion_jacobian(0, 1) = -s;
    motion_jacobian(1, 0) = s;
    motion_jacobian(1, 1) = c;

    Eigen::Matrix3d result =
        state_jacobian * state_covariance * state_jacobian.transpose() +
        motion_jacobian * motion_covariance * motion_jacobian.transpose();
    result = 0.5 * (result + result.transpose());
    for (int i = 0; i < 3; ++i) {
        result(i, i) = std::max(0.0, result(i, i));
    }
    return result;
}

}  // namespace gpu_amcl_cpp::localization_math

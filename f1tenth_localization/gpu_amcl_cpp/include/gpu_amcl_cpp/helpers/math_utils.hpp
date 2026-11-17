#pragma once

#include <cmath>
#include <limits>
#include <array>
#include <Eigen/Core>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

namespace gpu_amcl_cpp {

/** Planar pose and angle operations for localization. */
namespace math_utils {

/** Wrap radians to [-pi, pi]. */
inline double normalize_angle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

/** Signed angular difference `a - b`, wrapped to [-pi, pi]. */
inline double angle_diff(double a, double b) {
    return normalize_angle(a - b);
}

/** Quaternion yaw about Z, in radians. */
inline double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q) {
    const double norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (norm_sq <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    const double inv_norm = 1.0 / std::sqrt(norm_sq);
    const double x = q.x * inv_norm;
    const double y = q.y * inv_norm;
    const double z = q.z * inv_norm;
    const double w = q.w * inv_norm;

    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

/** Z-axis quaternion from yaw in radians. */
inline geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw) {
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(yaw / 2.0);
    q.w = std::cos(yaw / 2.0);
    return q;
}

/** Planar vector [x, y, yaw] from a ROS pose. */
inline Eigen::Vector3d pose_to_vec(const geometry_msgs::msg::Pose& p) {
    return {p.position.x, p.position.y, quaternion_to_yaw(p.orientation)};
}

/** ROS pose from [x, y, yaw], with z=0. */
inline geometry_msgs::msg::Pose vec_to_pose(const Eigen::Vector3d& v) {
    geometry_msgs::msg::Pose p;
    p.position.x = v[0];
    p.position.y = v[1];
    p.position.z = 0.0;
    p.orientation = yaw_to_quaternion(v[2]);
    return p;
}

/** Compose `a` with `delta` expressed in a's frame: T_ab * T_bc = T_ac.
 * Poses use [x, y, yaw]. */
inline Eigen::Vector3d se2_compose(const Eigen::Vector3d& a,
                                   const Eigen::Vector3d& delta) {
    double c = std::cos(a[2]);
    double s = std::sin(a[2]);
    return {
        a[0] + delta[0] * c - delta[1] * s,
        a[1] + delta[0] * s + delta[1] * c,
        normalize_angle(a[2] + delta[2])
    };
}

/** Inverse planar transform: T_ab -> T_ba, in [x, y, yaw] form. */
inline Eigen::Vector3d se2_inverse(const Eigen::Vector3d& pose) {
    double c = std::cos(pose[2]);
    double s = std::sin(pose[2]);
    return {
        -(pose[0] * c + pose[1] * s),
        pose[0] * s - pose[1] * c,
        normalize_angle(-pose[2])
    };
}

/** Relative transform `delta` in the `from` frame, such that
 * `se2_compose(from, delta)` equals `to`. Poses use [x, y, yaw]. */
inline Eigen::Vector3d se2_relative(const Eigen::Vector3d& from,
                                    const Eigen::Vector3d& to) {
    return se2_compose(se2_inverse(from), {to[0], to[1], to[2]});
}

/** Circular mean of `n` angles in radians; weights are normalized. */
inline double weighted_circular_mean(const double* angles,
                                     const double* weights,
                                     int n) {
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        s += weights[i] * std::sin(angles[i]);
        c += weights[i] * std::cos(angles[i]);
    }
    return std::atan2(s, c);
}

}  // namespace math_utils
}  // namespace gpu_amcl_cpp

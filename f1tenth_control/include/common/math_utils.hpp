#ifndef F1TENTH_CONTROL_MATH_UTILS_HPP_
#define F1TENTH_CONTROL_MATH_UTILS_HPP_

/** Geometry utilities; angles are radians and positions are metres.
 * Callers supply finite inputs. Median filtering is implemented in math_utils.cpp. */

#include "common/types.hpp"
#include <cmath>
#include <vector>

namespace f1tenth_control {
namespace math {

// Core math utilities

/** Wrap radians to [-PI, PI). */
inline double normalizeAngle(double angle) {
    angle = std::fmod(angle + constants::PI, constants::TWO_PI);
    return angle < 0 ? angle + constants::PI : angle - constants::PI;
}


inline constexpr double lerp(double a, double b, double t) {
    return a + t * (b - a);
}


inline double distance(const Point2D& a, const Point2D& b) {
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    return std::sqrt(dx * dx + dy * dy);
}


inline double distance(double x1, double y1, double x2, double y2) {
    return std::hypot(x2 - x1, y2 - y1);
}

/** Transform a robot-frame point into the global/map frame. */
inline Point2D localToGlobal(const Point2D& local_point, const Pose2D& robot_pose) {
    const double cos_theta = std::cos(robot_pose.theta);
    const double sin_theta = std::sin(robot_pose.theta);
    return {
        robot_pose.x + local_point.x * cos_theta - local_point.y * sin_theta,
        robot_pose.y + local_point.x * sin_theta + local_point.y * cos_theta
    };
}


std::vector<double> medianFilter(const std::vector<double>& data, size_t window_size);

}  // namespace math
}  // namespace f1tenth_control

#endif  // F1TENTH_CONTROL_MATH_UTILS_HPP_

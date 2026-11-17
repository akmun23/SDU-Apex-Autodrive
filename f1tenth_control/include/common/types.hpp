#ifndef F1TENTH_CONTROL_TYPES_HPP_
#define F1TENTH_CONTROL_TYPES_HPP_

/** Shared geometry and scan data; angles are radians and positions are metres. */

#include <vector>
#include <cmath>
#include <limits>

namespace f1tenth_control {


struct Point2D {
    double x{0.0};
    double y{0.0};

    Point2D() = default;
    
    Point2D(double x_, double y_) : x(x_), y(y_) {}
    
    double norm() const { return std::sqrt(x * x + y * y); }

    double dot(const Point2D& other) const { return x * other.x + y * other.y; }
    
    Point2D operator+(const Point2D& other) const { return {x + other.x, y + other.y}; }

    Point2D operator-(const Point2D& other) const { return {x - other.x, y - other.y}; }

    Point2D operator*(double scalar) const { return {x * scalar, y * scalar}; }
};


struct Pose2D {
    double x{0.0};
    double y{0.0};
    double theta{0.0};  // Heading in radians

    Pose2D() = default;
    
    Pose2D(double x_, double y_, double theta_) : x(x_), y(y_), theta(theta_) {}
    
    Point2D position() const { return {x, y}; }
};


struct VehicleState {
    Pose2D pose;
    double velocity{0.0};           // Linear velocity (m/s)
    double angular_velocity{0.0};   // Angular velocity (rad/s)
    double steering_angle{0.0};     // Current steering angle (rad)
};


struct DriveCommand {
    double speed{0.0};          // Target speed (m/s)
    double steering_angle{0.0}; // Target steering angle (rad)

    DriveCommand() = default;
    
    DriveCommand(double speed_, double angle_) : speed(speed_), steering_angle(angle_) {}
};


struct PolarPoint {
    double range{0.0};    // Distance in meters
    double angle{0.0};    // Angle in radians
    bool valid{true};

    PolarPoint() = default;

    PolarPoint(double r, double a, bool v = true) : range(r), angle(a), valid(v) {}
    
    Point2D toCartesian() const {
        return {range * std::cos(angle), range * std::sin(angle)};
    }
};


struct Gap {
    size_t start_idx{0};
    size_t end_idx{0};
    double start_angle{0.0};    // Start angle (rad)
    double end_angle{0.0};      // End angle (rad)
    double min_range{0.0};
    double max_range{0.0};
    double avg_range{0.0};
    size_t deepest_idx{0};
    double deepest_range{0.0};
    double angular_width{0.0};  // Angular width (rad)
    
    bool isValid() const { return end_idx > start_idx && angular_width > 0.0; }
    
    double centerAngle() const { return (start_angle + end_angle) / 2.0; }
};


struct ProcessedScan {
    std::vector<double> ranges{};
    std::vector<double> filtered_ranges;
    std::vector<double> angles;
    std::vector<bool> valid;
    std::vector<bool> disparity_blocked;    // Shadow of a closer obstacle.

    std::vector<bool> bubble_blocked;       // Minimum-clearance bubble.
                                            
    double angle_min{0.0};
    double angle_max{0.0};
    double angle_increment{0.0};
    double range_min{0.0};
    double range_max{0.0};
    
    size_t size() const { return ranges.size(); }
};


struct BoundaryPoint {
    Point2D position = Point2D(0.0, 0.0);   // Position in map frame
    double timestamp{0.0};
    bool is_left_wall{false};
    double confidence{1.0};
};


struct TrajectoryPoint {
    double x{0.0};                                                  // [m] X position
    double y{0.0};                                                  // [m] Y position
    double heading{0.0};                                            // [rad] Heading angle
    double velocity{0.0};                                           // [m/s] Target velocity
    double acceleration{0.0};                                       // [m/s^2] Profile acceleration
    double curvature{0.0};                                          // [1/m] Path curvature
    double arc_length{0.0};                                         // [m] Distance along path
    double left_bound{std::numeric_limits<double>::infinity()};     // [m] Left corridor half-width
    double right_bound{std::numeric_limits<double>::infinity()};    // [m] Right corridor half-width

    TrajectoryPoint() = default;
    
    TrajectoryPoint(double x_, double y_, double h, double v, double k, double s,
                    double l = std::numeric_limits<double>::infinity(),
                    double r = std::numeric_limits<double>::infinity())
        : x(x_), y(y_), heading(h), velocity(v), curvature(k), arc_length(s),
          left_bound(l), right_bound(r) {}
};


namespace constants {
    constexpr double PI = 3.14159265358979323846;
    constexpr double TWO_PI = 2.0 * PI;
    constexpr double DEG_TO_RAD = PI / 180.0;
    constexpr double RAD_TO_DEG = 180.0 / PI;
    constexpr double INF = std::numeric_limits<double>::infinity();
    constexpr double EPSILON = 1e-9;
}

}  // namespace f1tenth_control

#endif  // F1TENTH_CONTROL_TYPES_HPP_

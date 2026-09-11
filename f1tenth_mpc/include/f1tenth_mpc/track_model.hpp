#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace f1tenth_mpc {

struct TrackPoint {
  double s{0.0};
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double curvature{0.0};
  double speed{0.0};
  double left_bound{0.5};
  double right_bound{0.5};
};

struct TrackProjection {
  bool valid{false};
  std::size_t index{0};
  double s{0.0};
  double lateral_error{0.0};
  double heading_error{0.0};
  double distance{0.0};
};

class TrackModel {
public:
  bool load_csv(const std::string &path, std::string *error = nullptr);
  bool empty() const { return points_.size() < 2; }
  std::size_t size() const { return points_.size(); }
  double length() const { return length_; }
  const std::vector<TrackPoint> &points() const { return points_; }

  TrackPoint sample(double s) const;
  TrackProjection project(double x, double y, double yaw) const;

private:
  std::vector<TrackPoint> points_;
  double length_{0.0};
};

}  // namespace f1tenth_mpc

#include "f1tenth_mpc/track_model.hpp"

#include "f1tenth_mpc/vehicle_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace f1tenth_mpc {
namespace {

bool parse_double(const std::string &text, double &value)
{
  try {
    std::size_t consumed = 0;
    value = std::stod(text, &consumed);
    return consumed == text.size() && std::isfinite(value);
  } catch (...) {
    return false;
  }
}

std::vector<std::string> split_csv(const std::string &line)
{
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

}  // namespace

bool TrackModel::load_csv(const std::string &path, std::string *error)
{
  points_.clear();
  length_ = 0.0;
  std::ifstream input(path);
  if (!input) {
    if (error) *error = "could not open trajectory: " + path;
    return false;
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto fields = split_csv(line);
    if (fields.size() < 4) continue;  // header or malformed row
    TrackPoint point;
    double values[9]{};
    const std::size_t count = std::min<std::size_t>(fields.size(), 9);
    bool valid = true;
    for (std::size_t i = 0; i < count; ++i) {
      if (!parse_double(fields[i], values[i])) {
        valid = false;
        break;
      }
    }
    if (!valid || count < 4) continue;
    point.s = values[0];
    point.x = values[1];
    point.y = values[2];
    point.yaw = values[3];
    point.curvature = count > 4 ? values[4] : 0.0;
    point.speed = count > 5 ? std::max(0.0, values[5]) : 0.0;
    point.left_bound = count > 7 ? std::max(0.0, values[7]) : 0.5;
    point.right_bound = count > 8 ? std::max(0.0, values[8]) : 0.5;
    points_.push_back(point);
  }
  if (points_.size() < 2) {
    if (error) *error = "trajectory contains fewer than two numeric points";
    return false;
  }
  std::sort(points_.begin(), points_.end(),
            [](const TrackPoint &a, const TrackPoint &b) { return a.s < b.s; });
  const auto &last = points_.back();
  const auto &first = points_.front();
  const double closing = std::hypot(last.x - first.x, last.y - first.y);
  length_ = std::max(last.s, first.s) + closing;
  if (!(length_ > 0.0)) {
    if (error) *error = "trajectory length is not positive";
    points_.clear();
    return false;
  }
  return true;
}

TrackPoint TrackModel::sample(double s) const
{
  if (points_.empty()) return {};
  if (length_ > 0.0) {
    s = std::fmod(s, length_);
    if (s < 0.0) s += length_;
  }
  auto upper = std::upper_bound(
    points_.begin(), points_.end(), s,
    [](double value, const TrackPoint &point) { return value < point.s; });
  if (upper == points_.begin()) return points_.front();
  if (upper == points_.end()) {
    const auto &a = points_.back();
    const auto &b = points_.front();
    const double span = length_ - a.s;
    const double alpha = span > 1e-9 ? (s - a.s) / span : 0.0;
    TrackPoint result = a;
    result.s = s;
    result.x = a.x + alpha * (b.x - a.x);
    result.y = a.y + alpha * (b.y - a.y);
    result.yaw = wrap_angle(a.yaw + alpha * wrap_angle(b.yaw - a.yaw));
    result.curvature = a.curvature + alpha * (b.curvature - a.curvature);
    result.speed = a.speed + alpha * (b.speed - a.speed);
    result.left_bound = a.left_bound + alpha * (b.left_bound - a.left_bound);
    result.right_bound = a.right_bound + alpha * (b.right_bound - a.right_bound);
    return result;
  }
  const auto &b = *upper;
  const auto &a = *(upper - 1);
  const double span = b.s - a.s;
  const double alpha = span > 1e-9 ? (s - a.s) / span : 0.0;
  TrackPoint result = a;
  result.s = s;
  result.x = a.x + alpha * (b.x - a.x);
  result.y = a.y + alpha * (b.y - a.y);
  result.yaw = wrap_angle(a.yaw + alpha * wrap_angle(b.yaw - a.yaw));
  result.curvature = a.curvature + alpha * (b.curvature - a.curvature);
  result.speed = a.speed + alpha * (b.speed - a.speed);
  result.left_bound = a.left_bound + alpha * (b.left_bound - a.left_bound);
  result.right_bound = a.right_bound + alpha * (b.right_bound - a.right_bound);
  return result;
}

TrackProjection TrackModel::project(const double x, const double y, const double yaw) const
{
  TrackProjection result;
  if (empty()) return result;
  double best_distance_sq = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < points_.size(); ++i) {
    const auto &a = points_[i];
    const auto &b = points_[(i + 1) % points_.size()];
    const double bx = b.x - a.x;
    const double by = b.y - a.y;
    const double denom = bx * bx + by * by;
    const double alpha = denom > 1e-12
      ? clamp(((x - a.x) * bx + (y - a.y) * by) / denom, 0.0, 1.0) : 0.0;
    const double px = a.x + alpha * bx;
    const double py = a.y + alpha * by;
    const double dx = x - px;
    const double dy = y - py;
    const double distance_sq = dx * dx + dy * dy;
    if (distance_sq >= best_distance_sq) continue;
    best_distance_sq = distance_sq;
    const double segment_s = i + 1 < points_.size()
      ? a.s + alpha * (b.s - a.s)
      : a.s + alpha * (length_ - a.s);
    const double tangent_x = denom > 1e-12 ? bx / std::sqrt(denom) : std::cos(a.yaw);
    const double tangent_y = denom > 1e-12 ? by / std::sqrt(denom) : std::sin(a.yaw);
    result.valid = true;
    result.index = i;
    result.s = segment_s;
    result.distance = std::sqrt(distance_sq);
    result.lateral_error = tangent_x * dy - tangent_y * dx;
    result.heading_error = wrap_angle(yaw - std::atan2(tangent_y, tangent_x));
  }
  return result;
}

}  // namespace f1tenth_mpc

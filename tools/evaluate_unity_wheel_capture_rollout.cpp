#include "unity_source_plant.hpp"
#include "unity_source_plant_c.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

bool g_apply_sticky_tire_constraints = true;
bool g_apply_immediate_sticky_constraints = true;

constexpr double kPi = 3.14159265358979323846;
constexpr double kWheelRadiusM = 0.059;
constexpr double kComOffsetX = -0.00468;
constexpr double kComOffsetY = 0.00008;
constexpr double kComOffsetZ = 0.06434;
constexpr double kWheelX[4] = {0.17468, 0.17468, -0.15532, -0.15532};
constexpr double kWheelY[4] = {
  0.11574111133813858, -0.11590110510587692,
  0.11792000383138657, -0.11807999759912491};
constexpr double kWheelZ[4] = {
  -0.004435138776898384, -0.004435138776898384,
  -0.014340000227093697, -0.014340000227093697};

struct Sample {
  double time{};
  int diagnostic_trial{};
  int diagnostic_stage{};
  double target_speed{};
  double target_steering{};
  double target_throttle{};
  double throttle{};
  bool intention_to_accelerate{};
  double steering{};
  double u{};
  double v{};
  double r{};
  double height{};
  double height_rate{};
  double roll{};
  double roll_rate{};
  double pitch{};
  double pitch_rate{};
  double physical_steering{};
  double rack_steering_after_actuator{};
  double acceleration_body_x{};
  double acceleration_body_z{};
  bool acceleration_valid{};
  std::array<double, 4> wheel_omega{};
  std::array<double, 4> low_forward_speed_time_s{};
  std::array<double, 4> low_side_speed_time_s{};
  std::array<double, 4> contact_forward_speed_mps{};
  std::array<double, 4> contact_lateral_speed_mps{};
  std::array<double, 4> source_wheel_bottom_forward_speed_mps{};
  std::array<double, 4> source_wheel_bottom_lateral_speed_mps{};
  std::array<bool, 4> source_wheel_bottom_valid{};
  std::array<std::array<double, 3>, 4> collider_offset_body_m{};
  std::array<std::array<double, 3>, 4> source_wheel_bottom_body_m{};
  std::array<std::array<double, 3>, 4> source_tire_longitudinal_body{};
  std::array<double, 4> forward_slip{};
  std::array<double, 4> sideways_slip{};
  std::array<double, 4> hit_force{};
  std::array<double, 4> wheel_travel_projection{};
  bool all_contacts{};
};

struct ErrorSeries {
  std::vector<double> values;
  void add(double value) { values.push_back(std::abs(value)); }
  double rmse(const std::vector<double> &signed_values) const
  {
    if (signed_values.empty()) return 0.0;
    double sum = 0.0;
    for (double value : signed_values) sum += value * value;
    return std::sqrt(sum / static_cast<double>(signed_values.size()));
  }
  double percentile(double p) const
  {
    if (values.empty()) return 0.0;
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(sorted.size())) - 1.0);
    return sorted[std::min(index, sorted.size() - 1)];
  }
  double maximum() const
  {
    return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
  }
};

struct Metric {
  ErrorSeries absolute;
  std::vector<double> signed_values;
  void add(double error)
  {
    absolute.add(error);
    signed_values.push_back(error);
  }
  void report(const char *label) const
  {
    std::cout << "  " << label << " n=" << signed_values.size()
              << " rmse=" << std::fixed << std::setprecision(6)
              << absolute.rmse(signed_values)
              << " p95_abs=" << absolute.percentile(0.95)
              << " max_abs=" << absolute.maximum() << '\n';
  }
};

struct TrialMetrics {
  Metric u;
  Metric v;
  Metric r;
  Metric wheel_omega;
  Metric forward_slip;
  Metric sideways_slip;
  Metric normal_load;
  Metric raw_suspension_force;
  Metric longitudinal_acceleration;
  Metric lateral_acceleration;
  Metric yaw_acceleration;
  Metric suspension_jounce_rate;
  std::size_t transitions{};
};

struct MotionMetrics {
  Metric u;
  Metric v;
  Metric r;
};

struct ErrorPeak {
  double absolute_error{-1.0};
  std::size_t start_row{};
  int start_stage{};
  double start_speed{};
  double end_speed{};
  double target_steering{};
  double target_throttle{};
};

struct MotionPeaks {
  ErrorPeak u;
  ErrorPeak v;
  ErrorPeak r;
};

struct YawStepPeak {
  double absolute_error{-1.0};
  std::size_t row{};
  double predicted_r{};
  double measured_r{};
  double predicted_yaw_acceleration{};
  double measured_yaw_acceleration{};
  std::array<double, 4> predicted_longitudinal_slip{};
  std::array<double, 4> measured_longitudinal_slip{};
  std::array<double, 4> predicted_lateral_slip{};
  std::array<double, 4> measured_lateral_slip{};
  std::array<double, 4> predicted_contact_forward_speed{};
  std::array<double, 4> predicted_contact_lateral_speed{};
  std::array<double, 4> predicted_sticky_contact_forward_speed{};
  std::array<double, 4> predicted_sticky_contact_lateral_speed{};
  std::array<std::array<double, 3>, 4> predicted_sticky_contact_position{};
  std::array<std::array<double, 3>, 4> predicted_sticky_tire_longitudinal{};
  std::array<bool, 4> predicted_sticky_forward_active{};
  std::array<bool, 4> predicted_sticky_side_active{};
  std::array<double, 4> predicted_tire_longitudinal_force{};
  std::array<double, 4> predicted_tire_lateral_force{};
  std::array<double, 4> predicted_normal_load{};
  std::array<double, 4> predicted_suspension_force{};
  std::array<double, 4> predicted_jounce{};
  std::array<double, 4> predicted_jounce_rate{};
  std::array<double, 4> measured_jounce_rate{};
};

void report_compact(const char *name, const Metric &metric)
{
  std::cout << name << '=' << std::fixed << std::setprecision(5)
            << metric.absolute.rmse(metric.signed_values) << '/'
            << metric.absolute.percentile(0.95) << '/'
            << metric.absolute.maximum();
}

double number(const std::vector<std::string> &fields,
              const std::unordered_map<std::string, std::size_t> &columns,
              const char *name)
{
  const auto it = columns.find(name);
  if (it == columns.end() || it->second >= fields.size())
    throw std::runtime_error(std::string("missing capture field: ") + name);
  char *end = nullptr;
  const double value = std::strtod(fields[it->second].c_str(), &end);
  if (end == fields[it->second].c_str() || *end != '\0' || !std::isfinite(value))
    throw std::runtime_error(std::string("invalid capture value: ") + name);
  return value;
}

double number_or(const std::vector<std::string> &fields,
                 const std::unordered_map<std::string, std::size_t> &columns,
                 const char *name, double fallback)
{
  if (columns.find(name) == columns.end()) return fallback;
  return number(fields, columns, name);
}

std::vector<std::string> split(const std::string &line)
{
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const auto comma = line.find(',', start);
    fields.emplace_back(line.substr(start,
        comma == std::string::npos ? comma : comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return fields;
}

void derive_tilt(const std::vector<std::string> &f,
                 const std::unordered_map<std::string, std::size_t> &c,
                 double *roll, double *pitch)
{
  // Convert Unity's (right, up, forward) frame into the plant's
  // (forward, left, up) frame, then extract Z-Y-X yaw/pitch/roll.
  const double x = number(f, c, "rotation_x");
  const double y = number(f, c, "rotation_y");
  const double z = number(f, c, "rotation_z");
  const double w = number(f, c, "rotation_w");
  const double r[3][3] = {
    {1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),
     2.0 * (x*z + y*w)},
    {2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z),
     2.0 * (y*z - x*w)},
    {2.0 * (x*z - y*w), 2.0 * (y*z + x*w),
     1.0 - 2.0 * (x*x + y*y)}};
  // C maps Unity basis vectors to plant coordinates: [z,-x,y].
  constexpr int permutation[3] = {2, 0, 1};
  constexpr double sign[3] = {1.0, -1.0, 1.0};
  double m[3][3]{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      m[i][j] = sign[i] * r[permutation[i]][permutation[j]] * sign[j];
  *pitch = std::asin(std::clamp(-m[2][0], -1.0, 1.0));
  *roll = std::atan2(m[2][1], m[2][2]);
}

Sample parse_sample(const std::vector<std::string> &f,
                    const std::unordered_map<std::string, std::size_t> &c)
{
  Sample s;
  s.time = number(f, c, "fixed_time");
  s.diagnostic_trial = static_cast<int>(number_or(
      f, c, "diagnostic_trial", -1.0));
  s.diagnostic_stage = static_cast<int>(number_or(
      f, c, "diagnostic_stage", -1.0));
  s.target_speed = number_or(f, c, "diagnostic_target_speed_mps", 0.0);
  s.target_steering = number_or(f, c, "diagnostic_target_steering_rad", 0.0);
  s.target_throttle = number_or(f, c, "diagnostic_target_throttle", 0.0);
  s.throttle = number(f, c, "requested_throttle");
  double max_drive_torque = 0.0;
  double max_brake_torque = 0.0;
  for (const char *wheel : {"fl", "fr", "rl", "rr"}) {
    max_drive_torque = std::max(max_drive_torque, std::abs(number(
        f, c, (std::string(wheel) + "_motor_torque").c_str())));
    max_brake_torque = std::max(max_brake_torque, std::abs(number(
        f, c, (std::string(wheel) + "_brake_torque").c_str())));
  }
  // PxVehicleNoDrive defines intention from the actual queued torque arrays,
  // not from the caller's normalized throttle value.
  s.intention_to_accelerate =
      max_drive_torque > 0.0 && max_brake_torque == 0.0;
  s.steering = number(f, c, "requested_steering");
  s.u = number(f, c, "velocity_body_z");
  s.v = -number(f, c, "velocity_body_x");
  s.r = -number(f, c, "angular_velocity_body_y");
  s.height = number(f, c, "position_world_y");
  // PxVehicleUpdate's carChassisLinVel and wheelBottomPos are COM-referenced;
  // the plant's vertical-velocity slot is therefore COM world velocity too.
  // Converting this to actor-origin velocity mixes reference points with u/v.
  s.height_rate = number(f, c, "velocity_world_y");
  derive_tilt(f, c, &s.roll, &s.pitch);
  s.roll_rate = -number(f, c, "angular_velocity_body_z");
  s.pitch_rate = number(f, c, "angular_velocity_body_x");
  // FixedUpdate captures CurrentSteeringAngle before Steer() advances the
  // rack, then records rack_angle_deg after Steer(). The plant step consumes
  // the requested command and must start from that pre-update feedback value.
  s.physical_steering = number(f, c, "previous_feedback_steering");
  s.rack_steering_after_actuator =
      -number(f, c, "rack_angle_deg") * kPi / 180.0;
  s.acceleration_body_x = number(f, c, "acceleration_body_dvdt_x");
  s.acceleration_body_z = number(f, c, "acceleration_body_dvdt_z");
  s.acceleration_valid = number(f, c, "body_acceleration_valid") > 0.5;
  const std::array<double, 3> world_velocity{
    number(f, c, "velocity_world_x"),
    number(f, c, "velocity_world_y"),
    number(f, c, "velocity_world_z")};
  const std::array<double, 3> world_angular_velocity{
    number(f, c, "angular_velocity_world_x"),
    number(f, c, "angular_velocity_world_y"),
    number(f, c, "angular_velocity_world_z")};
  const std::array<double, 3> world_com{
    number(f, c, "center_of_mass_world_x"),
    number(f, c, "center_of_mass_world_y"),
    number(f, c, "center_of_mass_world_z")};
  const double qx = number(f, c, "rotation_x");
  const double qy = number(f, c, "rotation_y");
  const double qz = number(f, c, "rotation_z");
  const double qw = number(f, c, "rotation_w");
  const double rotation[3][3] = {
    {1.0 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qz*qw),
     2.0 * (qx*qz + qy*qw)},
    {2.0 * (qx*qy + qz*qw), 1.0 - 2.0 * (qx*qx + qz*qz),
     2.0 * (qy*qz - qx*qw)},
    {2.0 * (qx*qz - qy*qw), 2.0 * (qy*qz + qx*qw),
     1.0 - 2.0 * (qx*qx + qy*qy)}};
  constexpr const char *prefix[4] = {"fl", "fr", "rl", "rr"};
  s.all_contacts = true;
  for (std::size_t i = 0; i < 4; ++i) {
    s.wheel_omega[i] = number(f, c,
        (std::string(prefix[i]) + "_rpm").c_str()) * (2.0 * kPi / 60.0);
    s.forward_slip[i] = number(f, c,
        (std::string(prefix[i]) + "_forward_slip").c_str());
    s.sideways_slip[i] = number(f, c,
        (std::string(prefix[i]) + "_sideways_slip").c_str());
    s.hit_force[i] = number(f, c,
        (std::string(prefix[i]) + "_hit_force").c_str());
    const std::string hit_point_prefix =
        std::string(prefix[i]) + "_hit_point_world_";
    const std::string hit_forward_prefix =
        std::string(prefix[i]) + "_hit_forward_world_";
    const std::string hit_sideways_prefix =
        std::string(prefix[i]) + "_hit_sideways_world_";
    const std::string hit_normal_prefix =
        std::string(prefix[i]) + "_hit_normal_world_";
    const std::string wheel_collider_prefix =
        std::string(prefix[i]) + "_collider_position_world_";
    const std::string suspension_axis_prefix =
        std::string(prefix[i]) + "_suspension_axis_world_";
    std::array<double, 3> hit_offset{}, hit_forward{}, hit_sideways{};
    std::array<double, 3> hit_point{}, hit_normal{}, collider_position{};
    std::array<double, 3> suspension_axis{};
    std::size_t axis_index = 0;
    for (const char axis : {'x', 'y', 'z'}) {
      hit_offset[axis_index] = number(f, c,
          (hit_point_prefix + axis).c_str()) - world_com[axis_index];
      hit_point[axis_index] = number(f, c,
          (hit_point_prefix + axis).c_str());
      hit_forward[axis_index] = number(f, c,
          (hit_forward_prefix + axis).c_str());
      hit_sideways[axis_index] = number(f, c,
          (hit_sideways_prefix + axis).c_str());
      hit_normal[axis_index] = number(f, c,
          (hit_normal_prefix + axis).c_str());
      collider_position[axis_index] = number(f, c,
          (wheel_collider_prefix + axis).c_str());
      suspension_axis[axis_index] = number(f, c,
          (suspension_axis_prefix + axis).c_str());
      ++axis_index;
    }
    const std::array<double, 3> contact_velocity{
      world_velocity[0] + world_angular_velocity[1] * hit_offset[2] -
          world_angular_velocity[2] * hit_offset[1],
      world_velocity[1] + world_angular_velocity[2] * hit_offset[0] -
          world_angular_velocity[0] * hit_offset[2],
      world_velocity[2] + world_angular_velocity[0] * hit_offset[1] -
          world_angular_velocity[1] * hit_offset[0]};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      s.contact_forward_speed_mps[i] +=
          contact_velocity[axis] * hit_forward[axis];
      s.contact_lateral_speed_mps[i] +=
          contact_velocity[axis] * hit_sideways[axis];
    }
    const std::array<double, 3> hit_forward_body_unity{
      rotation[0][0] * hit_forward[0] + rotation[1][0] * hit_forward[1] +
          rotation[2][0] * hit_forward[2],
      rotation[0][1] * hit_forward[0] + rotation[1][1] * hit_forward[1] +
          rotation[2][1] * hit_forward[2],
      rotation[0][2] * hit_forward[0] + rotation[1][2] * hit_forward[1] +
          rotation[2][2] * hit_forward[2]};
    s.source_tire_longitudinal_body[i] = {
      hit_forward_body_unity[2], -hit_forward_body_unity[0],
      hit_forward_body_unity[1]};
    // PxVehicleUpdate::intersectRayPlane computes the wheel-bottom point by
    // intersecting the suspension ray with the cached hit plane. Sticky-tire
    // activation uses velocity at this wheel-bottom point, not at WheelHit's
    // ground point. Reconstruct the source quantity from the capture fields
    // so a threshold miss is not diagnosed from the wrong lever arm.
    if (number(f, c,
          (std::string(prefix[i]) + "_has_hit").c_str()) > 0.5) {
      const double radius = number(f, c,
          (std::string(prefix[i]) + "_radius").c_str());
      constexpr double max_compression_m = 0.025;
      // The source prefab serializes WheelCollider.center=(0, 0.025, 0).
      // Capture logs Transform.position separately, so shift along the
      // collider's recorded local-up/suspension axis before reconstructing
      // PhysX's suspension ray. Omitting this camber-rotated center offset
      // moves the front ray laterally by about 2.18 mm.
      constexpr double wheel_collider_center_y_m = 0.025;
      std::array<double, 3> ray_origin{}, ray_direction{};
      double numerator = 0.0, denominator = 0.0;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        ray_direction[axis] = -suspension_axis[axis];
        ray_origin[axis] = collider_position[axis] + suspension_axis[axis] *
            (wheel_collider_center_y_m + radius + max_compression_m);
        numerator += hit_normal[axis] *
            (hit_point[axis] - ray_origin[axis]);
        denominator += hit_normal[axis] * ray_direction[axis];
      }
      if (std::abs(denominator) > 1.0e-8) {
        const double ray_distance = numerator / denominator;
        std::array<double, 3> lever{}, wheel_bottom_velocity{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
          lever[axis] = ray_origin[axis] +
              ray_direction[axis] * ray_distance - world_com[axis];
        }
        wheel_bottom_velocity = {
          world_velocity[0] + world_angular_velocity[1] * lever[2] -
              world_angular_velocity[2] * lever[1],
          world_velocity[1] + world_angular_velocity[2] * lever[0] -
              world_angular_velocity[0] * lever[2],
          world_velocity[2] + world_angular_velocity[0] * lever[1] -
              world_angular_velocity[1] * lever[0]};
        for (std::size_t axis = 0; axis < 3; ++axis) {
          s.source_wheel_bottom_forward_speed_mps[i] +=
              wheel_bottom_velocity[axis] * hit_forward[axis];
          s.source_wheel_bottom_lateral_speed_mps[i] +=
              wheel_bottom_velocity[axis] * hit_sideways[axis];
        }
    // Convert both the collider mount and source wheel-bottom point into the
    // model frame (forward, left, up), relative to the Rigidbody COM. This
    // exposes whether a threshold disagreement is caused by using a visual
    // moving wheel center instead of the WheelCollider ray origin.
    const auto world_to_model = [&](const std::array<double, 3> &world_delta) {
      std::array<double, 3> unity_local{};
      for (std::size_t local_axis = 0; local_axis < 3; ++local_axis) {
        for (std::size_t world_axis = 0; world_axis < 3; ++world_axis)
          unity_local[local_axis] +=
              rotation[world_axis][local_axis] * world_delta[world_axis];
      }
      return std::array<double, 3>{unity_local[2], -unity_local[0],
                                   unity_local[1]};
    };
    std::array<double, 3> collider_delta{}, bottom_delta{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      collider_delta[axis] = collider_position[axis] - world_com[axis];
      bottom_delta[axis] = ray_origin[axis] +
          ray_direction[axis] * ray_distance - world_com[axis];
    }
    s.collider_offset_body_m[i] = world_to_model(collider_delta);
    s.source_wheel_bottom_body_m[i] = world_to_model(bottom_delta);
        s.source_wheel_bottom_valid[i] = true;
      }
    }
    const std::string wheel_pose_prefix =
        std::string(prefix[i]) + "_wheel_pose_world_";
    double travel = 0.0;
    for (const char axis : {'x', 'y', 'z'}) {
      const double wheel_pose = number(f, c,
          (wheel_pose_prefix + axis).c_str());
      const double collider_position = number(f, c,
          (wheel_collider_prefix + axis).c_str());
      const double suspension_axis = number(f, c,
          (suspension_axis_prefix + axis).c_str());
      travel += (wheel_pose - collider_position) * suspension_axis;
    }
    s.wheel_travel_projection[i] = travel;
    s.all_contacts = s.all_contacts && number(f, c,
        (std::string(prefix[i]) + "_has_hit").c_str()) > 0.5;
  }
  return s;
}

f1tenth_mpc::UnitySourcePlantState make_state(const Sample &sample)
{
  f1tenth_mpc::UnitySourcePlantState s{};
  s.u_mps = sample.u;
  s.v_mps = sample.v;
  s.yaw_rate_radps = sample.r;
  s.wheel_omega_radps = sample.wheel_omega;
  s.low_forward_speed_time_s = sample.low_forward_speed_time_s;
  s.low_side_speed_time_s = sample.low_side_speed_time_s;
  s.body_height_m = sample.height;
  s.body_height_rate_mps = sample.height_rate;
  s.roll_rad = sample.roll;
  s.roll_rate_radps = sample.roll_rate;
  s.pitch_rad = sample.pitch;
  s.pitch_rate_radps = sample.pitch_rate;
  s.physical_steering_rad = sample.physical_steering;

  // Seed damper history from the same ray/plane geometry used by the plant;
  // this prevents an artificial first-substep spring transient.
  const double sr = std::sin(s.roll_rad), cr = std::cos(s.roll_rad);
  const double sp = std::sin(s.pitch_rad), cp = std::cos(s.pitch_rad);
  const double normal[3] = {-sp, cp * sr, cp * cr};
  for (std::size_t i = 0; i < 4; ++i) {
    const double ray_start[3] = {
      kWheelX[i], kWheelY[i], kWheelZ[i] + kWheelRadiusM + 0.025};
    const double com_offset[3] = {kComOffsetX, kComOffsetY, kComOffsetZ};
    const double numerator = s.body_height_m +
      normal[0] * com_offset[0] + normal[1] * com_offset[1] +
      normal[2] * com_offset[2] + normal[0] * ray_start[0] +
      normal[1] * ray_start[1] + normal[2] * ray_start[2];
    const double ray_travel = numerator / normal[2];
    s.suspension_jounce_m[i] = std::clamp(
        2.0 * kWheelRadiusM + 0.025 - ray_travel, -0.025, 0.025);
  }
  s.suspension_history_initialized = true;
  return s;
}

void reconstruct_sticky_tire_history(std::vector<Sample> *samples)
{
  if (!samples || samples->empty()) return;
  std::array<double, 4> low_forward_time_s{};
  std::array<double, 4> low_side_time_s{};
  constexpr double kLowSpeedThresholdMps = 0.2;
  constexpr double kWheelRadiusM = 0.059;
  for (std::size_t row = 0; row < samples->size(); ++row) {
    Sample &sample = (*samples)[row];
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      sample.low_forward_speed_time_s[wheel] = low_forward_time_s[wheel];
      sample.low_side_speed_time_s[wheel] = low_side_time_s[wheel];
    }
    if (row + 1 >= samples->size()) break;
    const Sample &next = (*samples)[row + 1];
    const double dt = next.time - sample.time;
    if (std::abs(dt - 0.001) > 0.00015) {
      low_forward_time_s.fill(0.0);
      low_side_time_s.fill(0.0);
      continue;
    }
    const bool intention_to_accelerate = sample.intention_to_accelerate;
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      // PxVehicleUpdate.cpp updates these histories only while filtered tire
      // load and surface friction are positive. The capture's contact force
      // is the directly recorded proxy for that gate on this static open
      // plane; the timer inputs themselves come from WheelHit directions,
      // Rigidbody velocity/angular velocity, and encoder RPM.
      if (!(sample.hit_force[wheel] > 0.0)) continue;
      if (sample.source_wheel_bottom_valid[wheel] &&
          std::abs(sample.source_wheel_bottom_forward_speed_mps[wheel]) <
              kLowSpeedThresholdMps &&
          std::abs(sample.wheel_omega[wheel]) <
              kLowSpeedThresholdMps / kWheelRadiusM &&
          !intention_to_accelerate) {
        low_forward_time_s[wheel] += dt;
      } else {
        low_forward_time_s[wheel] = 0.0;
      }
      // PxVehicleUpdate::updateLowSideSpeedTimer resets this timer while the
      // driver intends to accelerate, like the forward timer. The separate
      // activation function ignores that flag; preserve both source behaviors.
      if (sample.source_wheel_bottom_valid[wheel] &&
          std::abs(sample.source_wheel_bottom_lateral_speed_mps[wheel]) <
              kLowSpeedThresholdMps &&
          !intention_to_accelerate) {
        low_side_time_s[wheel] += dt;
      } else {
        low_side_time_s[wheel] = 0.0;
      }
    }
  }
}

void accumulate(const Sample &truth, const f1tenth_mpc::UnitySourcePlantState &p,
                std::array<Metric, 10> *metrics)
{
  (*metrics)[0].add(p.u_mps - truth.u);
  (*metrics)[1].add(p.v_mps - truth.v);
  (*metrics)[2].add(p.yaw_rate_radps - truth.r);
  (*metrics)[3].add(p.body_height_m - truth.height);
  (*metrics)[4].add(p.roll_rad - truth.roll);
  (*metrics)[5].add(p.pitch_rad - truth.pitch);
  for (std::size_t i = 0; i < 4; ++i)
    (*metrics)[6 + i].add(p.wheel_omega_radps[i] - truth.wheel_omega[i]);
}

}  // namespace

int main(int argc, char **argv)
{
  const std::string ablation = argc == 3 ? argv[2] : "";
  if (argc < 2 || argc > 3 ||
      (argc == 3 && ablation != "--without-sticky" &&
       ablation != "--without-immediate-sticky")) {
    std::cerr << "usage: " << argv[0]
              << " UNITY_WHEEL_DYNAMICS.csv [--without-sticky|"
                 "--without-immediate-sticky]\n";
    return 2;
  }
  g_apply_sticky_tire_constraints = ablation != "--without-sticky";
  g_apply_immediate_sticky_constraints =
      ablation != "--without-immediate-sticky" &&
      ablation != "--without-sticky";
  const char *inspect_row_text = std::getenv("APEX_INSPECT_CAPTURE_ROW");
  const std::size_t inspect_row = inspect_row_text
      ? static_cast<std::size_t>(std::strtoull(inspect_row_text, nullptr, 10))
      : std::numeric_limits<std::size_t>::max();
  std::ifstream input(argv[1]);
  if (!input) {
    std::cerr << "cannot open capture: " << argv[1] << '\n';
    return 2;
  }
  std::string line;
  std::unordered_map<std::string, std::size_t> columns;
  std::size_t expected_fields = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto header = split(line);
    expected_fields = header.size();
    for (std::size_t i = 0; i < header.size(); ++i) columns.emplace(header[i], i);
    break;
  }
  std::vector<Sample> samples;
  std::size_t truncated_final_rows = 0;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto fields = split(line);
    if (fields.size() < expected_fields) {
      if (input.peek() == std::char_traits<char>::eof()) {
        ++truncated_final_rows;
        break;
      }
      throw std::runtime_error("incomplete non-final capture row");
    }
    samples.push_back(parse_sample(fields, columns));
  }
  if (samples.size() < 2) {
    std::cerr << "capture has fewer than two samples\n";
    return 2;
  }

  reconstruct_sticky_tire_history(&samples);

  std::array<Metric, 10> one_ms{};
  std::array<Metric, 10> held_25ms{};
  constexpr std::array<std::size_t, 5> kHorizonSteps{25, 100, 250, 500, 750};
  std::array<std::array<Metric, 10>, kHorizonSteps.size()> command_sequence{};
  std::array<std::map<int, MotionMetrics>, kHorizonSteps.size()>
      command_sequence_by_trial{};
  std::array<std::map<int, MotionPeaks>, kHorizonSteps.size()>
      command_sequence_peaks{};
  std::map<int, TrialMetrics> one_step_by_trial;
  std::array<std::map<int, TrialMetrics>, 5> one_step_by_trial_stage{};
  std::map<int, YawStepPeak> worst_one_step_yaw_by_trial;
  std::array<std::map<int, YawStepPeak>, 5>
      worst_one_step_yaw_by_trial_stage{};
  std::array<TrialMetrics, 5> one_step_by_stage{};
  std::array<std::size_t, kHorizonSteps.size()> command_sequence_count{};
  std::size_t one_ms_count = 0, held_count = 0, invalid_dt = 0;
  for (std::size_t i = 0; i + 1 < samples.size(); ++i) {
    const auto &current = samples[i];
    const auto &next = samples[i + 1];
    if (std::abs((next.time - current.time) - 0.001) > 0.00015) {
      ++invalid_dt;
      continue;
    }
    auto state = make_state(current);
    const f1tenth_mpc::UnitySourcePlantInput plant_input{
      f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
          current.steering, UNITY_SOURCE_MAX_STEERING_RAD),
      current.throttle, g_apply_sticky_tire_constraints,
      g_apply_immediate_sticky_constraints};
    f1tenth_mpc::UnitySourcePlantOutput output{};
    if (!f1tenth_mpc::unity_source_plant_step_1ms(state, plant_input, &output))
      continue;
    if (i >= inspect_row && i <= inspect_row + 2) {
      std::cout << "inspect row=" << i << " t=" << current.time
                << " actual_u/r=" << current.u << '/' << current.r
                << "->" << next.u << '/' << next.r
                << " pred_u/r=" << output.state.u_mps << '/'
                << output.state.yaw_rate_radps << " yaw_accel="
                << output.yaw_acceleration_radps2 << " steer_pred/act="
                << output.state.physical_steering_rad << '/'
                << current.rack_steering_after_actuator << " sticky=";
      for (std::size_t wheel = 0; wheel < 4; ++wheel)
        std::cout << output.sticky_forward_active[wheel] << ',';
      std::cout << " tireFy=";
      for (double force : output.tire_lateral_force_n)
        std::cout << force << ',';
      std::cout << " tireFx=";
      for (double force : output.tire_longitudinal_force_n)
        std::cout << force << ',';
      std::cout << " stickyV=";
      for (double speed : output.sticky_contact_forward_speed_mps)
        std::cout << speed << ',';
      std::cout << " timers=";
      for (double timer : state.low_forward_speed_time_s)
        std::cout << timer << ',';
      std::cout << " source_bottom=";
      for (double speed : current.source_wheel_bottom_forward_speed_mps)
        std::cout << speed << ',';
      std::cout << " next_slip=";
      for (double slip : next.forward_slip) std::cout << slip << ',';
      std::cout << '\n';
    }
    accumulate(next, output.state, &one_ms);
    const double yaw_error = output.state.yaw_rate_radps - next.r;
    TrialMetrics &trial_metrics = one_step_by_trial[current.diagnostic_trial];
    const std::size_t stage_index = static_cast<std::size_t>(
        std::clamp(current.diagnostic_stage, 0, 4));
    const auto update_yaw_peak = [&](YawStepPeak *yaw_peak) {
      if (std::abs(yaw_error) <= yaw_peak->absolute_error) return;
      yaw_peak->absolute_error = std::abs(yaw_error);
      yaw_peak->row = i;
      yaw_peak->predicted_r = output.state.yaw_rate_radps;
      yaw_peak->measured_r = next.r;
      yaw_peak->predicted_yaw_acceleration = output.yaw_acceleration_radps2;
      yaw_peak->measured_yaw_acceleration =
          (next.r - current.r) / (next.time - current.time);
      yaw_peak->predicted_longitudinal_slip = output.longitudinal_slip;
      yaw_peak->measured_longitudinal_slip = next.forward_slip;
      yaw_peak->predicted_lateral_slip = output.lateral_slip;
      yaw_peak->measured_lateral_slip = next.sideways_slip;
      yaw_peak->predicted_contact_forward_speed =
          output.contact_forward_speed_mps;
      yaw_peak->predicted_contact_lateral_speed =
          output.contact_lateral_speed_mps;
      yaw_peak->predicted_sticky_contact_forward_speed =
          output.sticky_contact_forward_speed_mps;
      yaw_peak->predicted_sticky_contact_lateral_speed =
          output.sticky_contact_lateral_speed_mps;
      yaw_peak->predicted_sticky_contact_position =
          output.sticky_contact_position_body_m;
      yaw_peak->predicted_sticky_tire_longitudinal =
          output.sticky_tire_longitudinal_body;
      yaw_peak->predicted_sticky_forward_active =
          output.sticky_forward_active;
      yaw_peak->predicted_sticky_side_active = output.sticky_side_active;
      yaw_peak->predicted_tire_longitudinal_force =
          output.tire_longitudinal_force_n;
      yaw_peak->predicted_tire_lateral_force = output.tire_lateral_force_n;
      yaw_peak->predicted_normal_load = output.normal_load_n;
      yaw_peak->predicted_suspension_force = output.suspension_force_n;
      yaw_peak->predicted_jounce = output.state.suspension_jounce_m;
      yaw_peak->predicted_jounce_rate = output.suspension_jounce_rate_mps;
      for (std::size_t wheel = 0; wheel < 4; ++wheel) {
        yaw_peak->measured_jounce_rate[wheel] =
            (next.wheel_travel_projection[wheel] -
             current.wheel_travel_projection[wheel]) /
            (next.time - current.time);
      }
    };
    update_yaw_peak(&worst_one_step_yaw_by_trial[current.diagnostic_trial]);
    update_yaw_peak(&worst_one_step_yaw_by_trial_stage[stage_index]
        [current.diagnostic_trial]);
    TrialMetrics &stage_metrics = one_step_by_stage[stage_index];
    TrialMetrics &trial_stage_metrics =
        one_step_by_trial_stage[stage_index][current.diagnostic_trial];
    const auto accumulate_transition = [&](TrialMetrics *metrics) {
      metrics->u.add(output.state.u_mps - next.u);
      metrics->v.add(output.state.v_mps - next.v);
      metrics->r.add(output.state.yaw_rate_radps - next.r);
      for (std::size_t wheel = 0; wheel < 4; ++wheel) {
        metrics->wheel_omega.add(output.state.wheel_omega_radps[wheel] -
                                 next.wheel_omega[wheel]);
        metrics->suspension_jounce_rate.add(
            output.suspension_jounce_rate_mps[wheel] -
            (next.wheel_travel_projection[wheel] -
             current.wheel_travel_projection[wheel]) /
                (next.time - current.time));
        if (next.all_contacts) {
          metrics->forward_slip.add(output.longitudinal_slip[wheel] -
                                    next.forward_slip[wheel]);
          metrics->sideways_slip.add(output.lateral_slip[wheel] -
                                     next.sideways_slip[wheel]);
          metrics->normal_load.add(output.normal_load_n[wheel] -
                                   next.hit_force[wheel]);
          metrics->raw_suspension_force.add(
              output.suspension_force_n[wheel] - next.hit_force[wheel]);
        }
      }
      if (next.acceleration_valid) {
        metrics->longitudinal_acceleration.add(
            output.longitudinal_acceleration_mps2 - next.acceleration_body_z);
        metrics->lateral_acceleration.add(
            output.lateral_acceleration_mps2 + next.acceleration_body_x);
        metrics->yaw_acceleration.add(output.yaw_acceleration_radps2 -
            (next.r - current.r) / (next.time - current.time));
      }
      ++metrics->transitions;
    };
    accumulate_transition(&trial_metrics);
    accumulate_transition(&stage_metrics);
    accumulate_transition(&trial_stage_metrics);
    ++one_ms_count;

    constexpr std::size_t kHeldSteps = 25;
    if (i + kHeldSteps >= samples.size()) continue;
    bool constant = true;
    for (std::size_t j = 1; j < kHeldSteps; ++j) {
      const auto &held = samples[i + j];
      if (std::abs((held.time - samples[i + j - 1].time) - 0.001) > 0.00015 ||
          held.throttle != current.throttle ||
          held.steering != current.steering || !held.all_contacts) {
        constant = false;
        break;
      }
    }
    if (!constant || !current.all_contacts || !samples[i + kHeldSteps].all_contacts)
      continue;
    state = make_state(current);
    bool valid = true;
    for (std::size_t j = 0; j < kHeldSteps; ++j) {
      if (!f1tenth_mpc::unity_source_plant_step_1ms(
              state, plant_input, &output)) {
        valid = false;
        break;
      }
      state = output.state;
    }
    if (!valid) continue;
    accumulate(samples[i + kHeldSteps], state, &held_25ms);
    ++held_count;
  }

  // MPC predicts forward under its planned command sequence. Test that mode
  // separately from the measured-state-conditioned one-step check: initialize
  // once from measured state, then consume only the captured command sequence
  // for the entire horizon without re-anchoring to future Unity states.
  constexpr std::size_t kControlGridStride = 25;
  for (std::size_t i = 0; i < samples.size(); i += kControlGridStride) {
    const auto &start = samples[i];
    for (std::size_t horizon_index = 0;
         horizon_index < kHorizonSteps.size(); ++horizon_index) {
      const std::size_t steps = kHorizonSteps[horizon_index];
      if (i + steps >= samples.size()) continue;
      bool valid_window = start.all_contacts &&
          samples[i + steps].all_contacts;
      for (std::size_t j = 0; valid_window && j < steps; ++j) {
        const auto &command = samples[i + j];
        const auto &next = samples[i + j + 1];
        valid_window = command.all_contacts &&
            command.diagnostic_trial == start.diagnostic_trial &&
            next.diagnostic_trial == start.diagnostic_trial &&
            std::abs((next.time - command.time) - 0.001) <= 0.00015;
      }
      if (!valid_window) continue;

      auto state = make_state(start);
      f1tenth_mpc::UnitySourcePlantOutput output{};
      bool valid = true;
      for (std::size_t j = 0; j < steps; ++j) {
        const auto &command = samples[i + j];
        const f1tenth_mpc::UnitySourcePlantInput plant_input{
          f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
              command.steering, UNITY_SOURCE_MAX_STEERING_RAD),
          command.throttle, g_apply_sticky_tire_constraints,
          g_apply_immediate_sticky_constraints};
        if (!f1tenth_mpc::unity_source_plant_step_1ms(
                state, plant_input, &output)) {
          valid = false;
          break;
        }
        state = output.state;
      }
      if (!valid) continue;
      accumulate(samples[i + steps], state,
                 &command_sequence[horizon_index]);
      MotionMetrics &by_trial = command_sequence_by_trial[horizon_index]
          [start.diagnostic_trial];
      by_trial.u.add(state.u_mps - samples[i + steps].u);
      by_trial.v.add(state.v_mps - samples[i + steps].v);
      by_trial.r.add(state.yaw_rate_radps - samples[i + steps].r);
      MotionPeaks &peaks = command_sequence_peaks[horizon_index]
          [start.diagnostic_trial];
      const auto record_peak = [&](ErrorPeak *peak, double error) {
        const double absolute_error = std::abs(error);
        if (absolute_error <= peak->absolute_error) return;
        peak->absolute_error = absolute_error;
        peak->start_row = i;
        peak->start_stage = start.diagnostic_stage;
        peak->start_speed = start.u;
        peak->end_speed = samples[i + steps].u;
        peak->target_steering = start.target_steering;
        peak->target_throttle = start.target_throttle;
      };
      record_peak(&peaks.u, state.u_mps - samples[i + steps].u);
      record_peak(&peaks.v, state.v_mps - samples[i + steps].v);
      record_peak(&peaks.r,
                  state.yaw_rate_radps - samples[i + steps].r);
      ++command_sequence_count[horizon_index];
    }
  }

  std::cout << "capture rows=" << samples.size() << " one_ms_pairs="
            << one_ms_count << " held_25ms_windows=" << held_count
            << " invalid_dt=" << invalid_dt
            << " truncated_final_rows=" << truncated_final_rows << '\n';
  const char *labels[10] = {
    "u_mps", "v_mps", "yaw_rate_radps", "body_height_m", "roll_rad",
    "pitch_rad", "wheel_omega_fl_radps", "wheel_omega_fr_radps",
    "wheel_omega_rl_radps", "wheel_omega_rr_radps"};
  std::cout << "one-millisecond measured-state-conditioned transition errors:\n";
  for (std::size_t i = 0; i < one_ms.size(); ++i) one_ms[i].report(labels[i]);
  std::cout << "one-millisecond per-trial diagnostics (RMSE/p95/max abs):\n";
  for (const auto &entry : one_step_by_trial) {
    const int trial = entry.first;
    const TrialMetrics &m = entry.second;
    const Sample *condition = nullptr;
    for (const auto &sample : samples) {
      if (sample.diagnostic_trial == trial) { condition = &sample; break; }
    }
    if (!condition) continue;
    std::cout << "  trial=" << trial << " target=" << condition->target_speed
              << "m/s," << condition->target_steering << "rad,q="
              << condition->target_throttle << " n=" << m.transitions << ' ';
    report_compact("u", m.u); std::cout << ' ';
    report_compact("v", m.v); std::cout << ' ';
    report_compact("r", m.r); std::cout << ' ';
    report_compact("wheel", m.wheel_omega); std::cout << ' ';
    report_compact("slipX", m.forward_slip); std::cout << ' ';
    report_compact("slipY", m.sideways_slip); std::cout << ' ';
    report_compact("loadN", m.normal_load); std::cout << ' ';
    report_compact("springN", m.raw_suspension_force); std::cout << ' ';
    report_compact("ax", m.longitudinal_acceleration); std::cout << ' ';
    report_compact("ay", m.lateral_acceleration); std::cout << ' ';
    report_compact("yawdd", m.yaw_acceleration); std::cout << '\n';
    report_compact("jounceRate", m.suspension_jounce_rate); std::cout << '\n';
  }
  std::cout << "one-millisecond per-stage aggregate (RMSE/p95/max abs):\n";
  for (std::size_t stage = 0; stage < one_step_by_stage.size(); ++stage) {
    const TrialMetrics &m = one_step_by_stage[stage];
    std::cout << "  stage=" << stage << " n=" << m.transitions << ' ';
    report_compact("u", m.u); std::cout << ' ';
    report_compact("v", m.v); std::cout << ' ';
    report_compact("r", m.r); std::cout << ' ';
    report_compact("slipX", m.forward_slip); std::cout << ' ';
    report_compact("slipY", m.sideways_slip); std::cout << ' ';
    report_compact("loadN", m.normal_load); std::cout << ' ';
    report_compact("springN", m.raw_suspension_force); std::cout << ' ';
    report_compact("ax", m.longitudinal_acceleration); std::cout << ' ';
    report_compact("ay", m.lateral_acceleration); std::cout << ' ';
    report_compact("yawdd", m.yaw_acceleration); std::cout << '\n';
  }
  std::cout << "one-millisecond per-trial/per-stage motion errors "
               "(RMSE/p95/max abs):\n";
  for (std::size_t stage = 0; stage < one_step_by_trial_stage.size(); ++stage) {
    for (const auto &entry : one_step_by_trial_stage[stage]) {
      const TrialMetrics &m = entry.second;
      std::cout << "  trial=" << entry.first << " stage=" << stage
                << " n=" << m.transitions << ' ';
      report_compact("u", m.u); std::cout << ' ';
      report_compact("v", m.v); std::cout << ' ';
      report_compact("r", m.r); std::cout << ' ';
      report_compact("yawdd", m.yaw_acceleration); std::cout << '\n';
      report_compact("jounceRate", m.suspension_jounce_rate); std::cout << '\n';
    }
  }
  std::cout << "largest one-step yaw residual per trial; truth wheel slips and "
               "plant wheel forces at that transition:\n";
  for (const auto &entry : worst_one_step_yaw_by_trial) {
    const int trial = entry.first;
    const YawStepPeak &peak = entry.second;
    const Sample &current = samples[peak.row];
    const Sample &next = samples[peak.row + 1];
    std::cout << "  trial=" << trial << " row=" << peak.row
              << " stage=" << current.diagnostic_stage
              << " q=" << current.throttle
              << " steer_cmd=" << current.steering
              << " steer_feedback_rad=" << current.physical_steering
              << " rack_after_actuator_rad="
              << current.rack_steering_after_actuator
              << " u=" << current.u << " v=" << current.v
              << " r=" << current.r
              << " yaw_rate_error=" << peak.absolute_error
              << " r_pred/truth=" << peak.predicted_r << '/'
              << peak.measured_r << " yaw_accel_pred/truth="
              << peak.predicted_yaw_acceleration << '/'
              << peak.measured_yaw_acceleration << "\n    wheel[fl,fr,rl,rr]"
              << std::setprecision(10);
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      std::cout << "\n      " << wheel
                << " slipX_pred/truth="
                << peak.predicted_longitudinal_slip[wheel] << '/'
                << peak.measured_longitudinal_slip[wheel]
                << " slipY_pred/truth="
                << peak.predicted_lateral_slip[wheel] << '/'
                << peak.measured_lateral_slip[wheel]
                << " contactVxy="
                << peak.predicted_contact_forward_speed[wheel] << ','
                << peak.predicted_contact_lateral_speed[wheel]
                << " stickyVxy="
                << peak.predicted_sticky_contact_forward_speed[wheel] << ','
                << peak.predicted_sticky_contact_lateral_speed[wheel]
                << " measuredVxy="
                << current.contact_forward_speed_mps[wheel] << ','
                << current.contact_lateral_speed_mps[wheel]
                << " sourceBottomVx="
                << current.source_wheel_bottom_forward_speed_mps[wheel]
                << " colliderOffsetBody="
                << current.collider_offset_body_m[wheel][0] << ','
                << current.collider_offset_body_m[wheel][1] << ','
                << current.collider_offset_body_m[wheel][2]
                << " sourceBottomBody="
                << current.source_wheel_bottom_body_m[wheel][0] << ','
                << current.source_wheel_bottom_body_m[wheel][1] << ','
                << current.source_wheel_bottom_body_m[wheel][2]
                << " predictedBottomDelta="
                << peak.predicted_sticky_contact_position[wheel][0] -
                    current.source_wheel_bottom_body_m[wheel][0] << ','
                << peak.predicted_sticky_contact_position[wheel][1] -
                    current.source_wheel_bottom_body_m[wheel][1] << ','
                << peak.predicted_sticky_contact_position[wheel][2] -
                    current.source_wheel_bottom_body_m[wheel][2]
                << " stickyDirBody="
                << peak.predicted_sticky_tire_longitudinal[wheel][0] << ','
                << peak.predicted_sticky_tire_longitudinal[wheel][1] << ','
                << peak.predicted_sticky_tire_longitudinal[wheel][2]
                << " sourceDirBody="
                << current.source_tire_longitudinal_body[wheel][0] << ','
                << current.source_tire_longitudinal_body[wheel][1] << ','
                << current.source_tire_longitudinal_body[wheel][2]
                << " sourceImmediate="
                << (current.source_wheel_bottom_valid[wheel] &&
                    current.throttle == 0.0 &&
                    std::abs(current.source_wheel_bottom_forward_speed_mps[wheel]) < 0.2 &&
                    current.wheel_omega[wheel] == 0.0)
                << " measuredImmediate="
                << (current.throttle == 0.0 &&
                    std::abs(current.contact_forward_speed_mps[wheel]) < 0.2 &&
                    current.wheel_omega[wheel] == 0.0)
                << " omega=" << current.wheel_omega[wheel]
                << " nextOmega=" << next.wheel_omega[wheel]
                << " nextSlipX=" << next.forward_slip[wheel]
                << " stickyTimerFS="
                << current.low_forward_speed_time_s[wheel] << ','
                << current.low_side_speed_time_s[wheel]
                << " stickyActive="
                << peak.predicted_sticky_forward_active[wheel] << ','
                << peak.predicted_sticky_side_active[wheel]
                << " forceXY=" << peak.predicted_tire_longitudinal_force[wheel]
                << ',' << peak.predicted_tire_lateral_force[wheel]
                << " load=" << peak.predicted_normal_load[wheel]
                << " springN=" << peak.predicted_suspension_force[wheel]
                << " hitN=" << next.hit_force[wheel];
    }
    std::cout << '\n';
  }
  std::cout << "largest one-step yaw residual in held-turn stage:\n";
  for (const int trial : {14, 15, 16}) {
    const auto &stage_peaks = worst_one_step_yaw_by_trial_stage[2];
    const auto entry = stage_peaks.find(trial);
    if (entry == stage_peaks.end()) continue;
    const YawStepPeak &peak = entry->second;
    const Sample &current = samples[peak.row];
    const Sample &next = samples[peak.row + 1];
    std::cout << "  trial=" << trial << " row=" << peak.row
              << " q=" << current.throttle << " steer_cmd="
              << current.steering << " steer_feedback_rad="
              << current.physical_steering << " rack_after_actuator_rad="
              << current.rack_steering_after_actuator
              << " roll/pitch=" << current.roll << '/' << current.pitch
              << " u/v/r=" << current.u << '/' << current.v << '/'
              << current.r << " yaw_rate_error=" << peak.absolute_error
              << " r_pred/truth=" << peak.predicted_r << '/'
              << peak.measured_r << " yaw_accel_pred/truth="
              << peak.predicted_yaw_acceleration << '/'
              << peak.measured_yaw_acceleration << '\n';
    for (std::size_t wheel = 0; wheel < 4; ++wheel) {
      std::cout << "    wheel=" << wheel << " slipX="
                << peak.predicted_longitudinal_slip[wheel] << '/'
                << peak.measured_longitudinal_slip[wheel] << " slipY="
                << peak.predicted_lateral_slip[wheel] << '/'
                << peak.measured_lateral_slip[wheel] << " forceXY="
                << peak.predicted_tire_longitudinal_force[wheel] << ','
                << peak.predicted_tire_lateral_force[wheel]
                << " filtered/raw/hitN=" << peak.predicted_normal_load[wheel]
                << '/' << peak.predicted_suspension_force[wheel] << '/'
                << next.hit_force[wheel] << " jounce/rate="
                << peak.predicted_jounce[wheel] << '/'
                << peak.predicted_jounce_rate[wheel] << "/meas="
                << peak.measured_jounce_rate[wheel] << '\n';
    }
  }
  std::cout << "25-millisecond held-command recursive rollout errors:\n";
  for (std::size_t i = 0; i < held_25ms.size(); ++i)
    held_25ms[i].report(labels[i]);
  std::cout << "open-loop captured-command sequence errors (measured initial "
               "state; no future-state correction):\n";
  for (std::size_t horizon = 0; horizon < kHorizonSteps.size(); ++horizon) {
    std::cout << "  horizon_ms=" << kHorizonSteps[horizon]
              << " windows=" << command_sequence_count[horizon] << '\n';
    for (std::size_t i = 0; i < command_sequence[horizon].size(); ++i)
      command_sequence[horizon][i].report(labels[i]);
  }
  std::cout << "command-sequence motion error by trial and horizon "
               "(RMSE/p95/max abs):\n";
  for (std::size_t horizon = 0; horizon < kHorizonSteps.size(); ++horizon) {
    for (const auto &entry : command_sequence_by_trial[horizon]) {
      const MotionMetrics &m = entry.second;
      std::cout << "  " << kHorizonSteps[horizon] << "ms trial="
                << entry.first << ' ';
      report_compact("u", m.u); std::cout << ' ';
      report_compact("v", m.v); std::cout << ' ';
      report_compact("r", m.r); std::cout << '\n';
    }
  }
  std::cout << "750 ms command-sequence error by trial (RMSE/p95/max abs):\n";
  for (const auto &entry : command_sequence_by_trial.back()) {
    const MotionMetrics &m = entry.second;
    std::cout << "  trial=" << entry.first << ' ';
    report_compact("u", m.u); std::cout << ' ';
    report_compact("v", m.v); std::cout << ' ';
    report_compact("r", m.r); std::cout << '\n';
  }
  std::cout << "750 ms largest endpoint-error window by trial (absolute error, "
               "start-row/stage/speed, endpoint-speed):\n";
  for (const auto &entry : command_sequence_peaks.back()) {
    const MotionPeaks &p = entry.second;
    const auto print_peak = [](const char *name, const ErrorPeak &peak) {
      std::cout << ' ' << name << '=' << std::fixed << std::setprecision(5)
                << peak.absolute_error << "@" << peak.start_row << "/s"
                << peak.start_stage << "/u" << peak.start_speed << "->u"
                << peak.end_speed;
    };
    std::cout << "  trial=" << entry.first;
    print_peak("u", p.u);
    print_peak("v", p.v);
    print_peak("r", p.r);
    std::cout << '\n';
  }
  const ErrorPeak *worst_yaw_peak = nullptr;
  int worst_yaw_trial = -1;
  for (const auto &entry : command_sequence_peaks.back()) {
    if (!worst_yaw_peak ||
        entry.second.r.absolute_error > worst_yaw_peak->absolute_error) {
      worst_yaw_peak = &entry.second.r;
      worst_yaw_trial = entry.first;
    }
  }
  if (worst_yaw_peak && worst_yaw_peak->absolute_error >= 0.0 &&
      worst_yaw_peak->start_row + kHorizonSteps.back() < samples.size()) {
    const std::size_t start_row = worst_yaw_peak->start_row;
    auto state = make_state(samples[start_row]);
    std::cout << "750 ms worst-yaw window trace; trial=" << worst_yaw_trial
              << " start_row=" << start_row
              << " (predictions use only the initial measured state and the "
                 "recorded command sequence):\n";
    std::cout << "  offset_ms stage q steer u_truth u_pred v_truth v_pred "
                 "r_truth r_pred wheel_omega_truth_mean "
                 "wheel_omega_pred_mean\n";
    for (std::size_t offset = 0; offset <= kHorizonSteps.back(); ++offset) {
      const Sample &truth = samples[start_row + offset];
      if (offset % 25 == 0 || offset >= 745) {
        double truth_wheel_mean = 0.0;
        double predicted_wheel_mean = 0.0;
        for (std::size_t wheel = 0; wheel < 4; ++wheel) {
          truth_wheel_mean += truth.wheel_omega[wheel] * 0.25;
          predicted_wheel_mean += state.wheel_omega_radps[wheel] * 0.25;
        }
        const Sample &command = samples[
            std::min(start_row + offset, samples.size() - 1)];
        std::cout << "  " << offset << ' ' << truth.diagnostic_stage << ' '
                  << command.throttle << ' ' << command.steering << ' '
                  << truth.u << ' ' << state.u_mps << ' ' << truth.v << ' '
                  << state.v_mps << ' ' << truth.r << ' '
                  << state.yaw_rate_radps << ' ' << truth_wheel_mean << ' '
                  << predicted_wheel_mean;
        if (offset >= 745) {
          std::cout << " timersF";
          for (double timer : state.low_forward_speed_time_s)
            std::cout << ' ' << timer;
          std::cout << " timersS";
          for (double timer : state.low_side_speed_time_s)
            std::cout << ' ' << timer;
        }
        std::cout << '\n';
      }
      if (offset == kHorizonSteps.back()) break;
      const Sample &command = samples[start_row + offset];
      const f1tenth_mpc::UnitySourcePlantInput plant_input{
        f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
            command.steering, UNITY_SOURCE_MAX_STEERING_RAD),
        command.throttle, g_apply_sticky_tire_constraints,
        g_apply_immediate_sticky_constraints};
      f1tenth_mpc::UnitySourcePlantOutput output{};
      if (!f1tenth_mpc::unity_source_plant_step_1ms(
              state, plant_input, &output)) break;
      state = output.state;
    }
  }
  return 0;
}

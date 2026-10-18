#include "f1tenth_localization/physx_odom_runtime.hpp"
#include "f1tenth_localization/physx_odom_actuation.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace f1tenth_localization
{
namespace
{

constexpr unsigned int kPhysicsStepsPerPacket =
  static_cast<unsigned int>(detail::kPhysicsStepsPerPacket);
constexpr unsigned int kInitialRestSettleSteps = 1000;
constexpr float kWheelbaseMm = 324.0f;
constexpr float kTrackWidthMm = 236.0f;
// The active scene overrides the prefab's 85.6 N m default with 428 N m.
// CAWD divides nonzero drive torque by four; CAWB applies full MotorTorque
// independently as brake torque to each wheel when throttle is exactly zero.
constexpr float kMotorTorqueNm = 428.0f;

float radians(float degrees)
{
  return degrees * (detail::kPi / 180.0f);
}

std::pair<float, float> unity_ackermann(float center_angle_deg)
{
  const float angle_rad = radians(center_angle_deg);
  const float tangent = std::tan(angle_rad);
  const float numerator = 2.0f * kWheelbaseMm * tangent;
  const float left = std::atan(
    numerator / (2.0f * kWheelbaseMm + kTrackWidthMm * tangent));
  const float right = std::atan(
    numerator / (2.0f * kWheelbaseMm - kTrackWidthMm * tangent));
  return {left, right};
}

void close_fd(int fd)
{
  if (fd >= 0) {
    close(fd);
  }
}

}  // namespace

PhysxOdomRuntime::PhysxOdomRuntime(
  const std::string & executable,
  const std::string & initial_state,
  const std::string & shape_audit)
{
  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (pipe(to_child) != 0 || pipe(from_child) != 0) {
    const int error = errno;
    close_fd(to_child[0]);
    close_fd(to_child[1]);
    close_fd(from_child[0]);
    close_fd(from_child[1]);
    throw std::runtime_error(
            "cannot create PhysX Odom pipes: " + std::string(std::strerror(error)));
  }

  std::signal(SIGPIPE, SIG_IGN);
  child_pid_ = fork();
  if (child_pid_ < 0) {
    const int error = errno;
    close_fd(to_child[0]);
    close_fd(to_child[1]);
    close_fd(from_child[0]);
    close_fd(from_child[1]);
    throw std::runtime_error(
            "cannot start PhysX Odom process: " + std::string(std::strerror(error)));
  }
  if (child_pid_ == 0) {
    (void)dup2(to_child[0], STDIN_FILENO);
    (void)dup2(from_child[1], STDOUT_FILENO);
    close_fd(to_child[0]);
    close_fd(to_child[1]);
    close_fd(from_child[0]);
    close_fd(from_child[1]);
    execl(
      executable.c_str(), executable.c_str(), "--server",
      initial_state.c_str(), shape_audit.c_str(), static_cast<char *>(nullptr));
    _exit(127);
  }

  close_fd(to_child[0]);
  close_fd(from_child[1]);
  child_input_ = fdopen(to_child[1], "w");
  child_output_ = fdopen(from_child[0], "r");
  if (child_input_ == nullptr || child_output_ == nullptr) {
    const int error = errno;
    if (child_input_ != nullptr) {
      std::fclose(child_input_);
      child_input_ = nullptr;
    } else {
      close_fd(to_child[1]);
    }
    if (child_output_ != nullptr) {
      std::fclose(child_output_);
      child_output_ = nullptr;
    } else {
      close_fd(from_child[0]);
    }
    (void)waitpid(child_pid_, nullptr, 0);
    child_pid_ = -1;
    throw std::runtime_error(
            "cannot open PhysX Odom process pipes: " + std::string(std::strerror(error)));
  }
  (void)setvbuf(child_input_, nullptr, _IOLBF, 0);

  // The reset pose is not a static suspension equilibrium. Let the source
  // plant settle on the plane before its first estimate is used; the normal
  // startup order starts this node before the simulator/vehicle begins moving.
  const std::array<float, kPhysicsStepsPerPacket> restSteering{};
  const unsigned int settlePackets = kInitialRestSettleSteps /
    kPhysicsStepsPerPacket;
  for (unsigned int index = 0; index < settlePackets; ++index) {
    (void)send_physics_packet(0.0f, restSteering);
  }
}

PhysxOdomRuntime::~PhysxOdomRuntime()
{
  if (child_input_ != nullptr) {
    (void)std::fprintf(child_input_, "{\"op\":\"shutdown\"}\n");
    (void)std::fflush(child_input_);
    std::fclose(child_input_);
    child_input_ = nullptr;
  }
  if (child_output_ != nullptr) {
    std::fclose(child_output_);
    child_output_ = nullptr;
  }
  if (child_pid_ > 0) {
    (void)waitpid(child_pid_, nullptr, 0);
    child_pid_ = -1;
  }
}

PredictedPlantState PhysxOdomRuntime::advance(
  double steering_command_norm, double throttle_command_norm,
  bool steering_feedback_valid, double steering_feedback_rad)
{
  if (!std::isfinite(steering_command_norm) ||
    !std::isfinite(throttle_command_norm))
  {
    throw std::runtime_error("non-finite actuator command sent to PhysX Odom");
  }
  const float steering_command = static_cast<float>(
    std::clamp(steering_command_norm, -1.0, 1.0));
  const float command_throttle = static_cast<float>(
    std::clamp(throttle_command_norm, -1.0, 1.0));
  const auto actuation = detail::make_physx_packet_actuation(
    center_steering_angle_deg_, delayed_steering_commands_norm_[0],
    delayed_throttle_commands_norm_[0], steering_feedback_valid,
    steering_feedback_rad);
  delayed_steering_commands_norm_[0] = delayed_steering_commands_norm_[1];
  delayed_steering_commands_norm_[1] = steering_command;
  delayed_throttle_commands_norm_[0] = delayed_throttle_commands_norm_[1];
  delayed_throttle_commands_norm_[1] = command_throttle;
  PredictedPlantState output = send_physics_packet(
    actuation.throttle_normalized, actuation.steering_angles_deg);
  for (std::size_t slot = 0; slot < 2; ++slot) {
    output.delayed_steering_commands_normalized[slot] =
      delayed_steering_commands_norm_[slot];
    output.delayed_throttle_commands_normalized[slot] =
      delayed_throttle_commands_norm_[slot];
  }
  return output;
}

void PhysxOdomRuntime::correct(
  double body_u_mps, double body_v_com_mps, double yaw_rate_radps,
  double yaw_rad, double roll_rad, double pitch_rad,
  double body_vertical_velocity_world_mps,
  double roll_rate_radps, double pitch_rate_radps,
  const std::array<double, 4> & wheel_omega_radps)
{
  if (child_input_ == nullptr || !std::isfinite(body_u_mps) ||
    !std::isfinite(body_v_com_mps) || !std::isfinite(yaw_rate_radps) ||
    !std::isfinite(yaw_rad) || !std::isfinite(roll_rad) ||
    !std::isfinite(pitch_rad) ||
    !std::isfinite(body_vertical_velocity_world_mps) ||
    !std::isfinite(roll_rate_radps) ||
    !std::isfinite(pitch_rate_radps) ||
    !std::all_of(wheel_omega_radps.begin(), wheel_omega_radps.end(),
      [](double value) {return std::isfinite(value);}))
  {
    throw std::runtime_error("invalid measured correction for PhysX Odom state");
  }
  const int written = std::fprintf(
    child_input_,
    "{\"op\":\"correct\",\"body_forward_velocity_mps\":%.12g,"
    "\"body_lateral_velocity_com_mps\":%.12g,\"yaw_rate_radps\":%.12g,"
    "\"yaw_rad\":%.12g,\"roll_rad\":%.12g,\"pitch_rad\":%.12g,"
    "\"body_vertical_velocity_world_mps\":%.12g,"
    "\"roll_rate_radps\":%.12g,\"pitch_rate_radps\":%.12g,"
    "\"wheel_omega_radps\":[%.12g,%.12g,%.12g,%.12g]}\n",
    body_u_mps, body_v_com_mps, yaw_rate_radps, yaw_rad, roll_rad,
    pitch_rad, body_vertical_velocity_world_mps,
    roll_rate_radps, pitch_rate_radps,
    wheel_omega_radps[0], wheel_omega_radps[1],
    wheel_omega_radps[2], wheel_omega_radps[3]);
  if (written < 0 || std::fflush(child_input_) != 0) {
    throw std::runtime_error("failed sending sensor correction to PhysX Odom");
  }
  char * line = nullptr;
  size_t capacity = 0;
  const ssize_t count = getline(&line, &capacity, child_output_);
  const bool acknowledged = count > 0 &&
    std::strstr(line, "\"op\":\"corrected\"") != nullptr;
  std::free(line);
  if (!acknowledged) {
    throw std::runtime_error("PhysX Odom did not acknowledge sensor correction");
  }
}

PredictedPlantState PhysxOdomRuntime::send_physics_packet(
  float throttle_input,
  const std::array<float, kPhysicsStepsPerPacket> & steering_angles_deg)
{
  if (child_input_ == nullptr) {
    throw std::runtime_error("PhysX Odom process is not available");
  }
  const float drive_torque = throttle_input == 0.0f ?
    0.0f : kMotorTorqueNm * throttle_input / 4.0f;
  const float brake_torque = throttle_input == 0.0f ? kMotorTorqueNm : 0.0f;
  std::array<char, 16384> request{};
  std::size_t offset = 0;
  const auto append = [&request, &offset](const char * format, auto... values) {
      if (offset >= request.size()) return false;
      const int count = std::snprintf(
        request.data() + offset, request.size() - offset, format, values...);
      if (count < 0 || static_cast<std::size_t>(count) >= request.size() - offset)
        return false;
      offset += static_cast<std::size_t>(count);
      return true;
    };
  const auto append_literal = [&request, &offset](const char * text,
                                                  std::size_t length) {
      if (offset >= request.size() || length >= request.size() - offset)
        return false;
      std::memcpy(request.data() + offset, text, length);
      offset += length;
      request[offset] = '\0';
      return true;
    };
  const double packet_start_s = static_cast<double>(physics_step_index_) * 0.001;
  bool valid = append(
    "{\"op\":\"rollout\",\"stages\":[{\"capture_row\":%llu,"
    "\"time_s\":%.9f,\"wheel_commands_by_tick\":[",
    static_cast<unsigned long long>(physics_step_index_), packet_start_s);
  for (std::size_t tick = 0; valid && tick < kPhysicsStepsPerPacket; ++tick) {
    const auto [front_left, front_right] =
      unity_ackermann(steering_angles_deg[tick]);
    if (tick > 0) valid = append_literal(",", 1);
    if (valid) valid = append(
      "[{\"steer_angle_rad\":%.9g,\"drive_torque_nm\":%.9g,"
      "\"brake_torque_nm\":%.9g},"
      "{\"steer_angle_rad\":%.9g,\"drive_torque_nm\":%.9g,"
      "\"brake_torque_nm\":%.9g},"
      "{\"steer_angle_rad\":0,\"drive_torque_nm\":%.9g,"
      "\"brake_torque_nm\":%.9g},"
      "{\"steer_angle_rad\":0,\"drive_torque_nm\":%.9g,"
      "\"brake_torque_nm\":%.9g}]",
      static_cast<double>(front_left), static_cast<double>(drive_torque),
      static_cast<double>(brake_torque), static_cast<double>(front_right),
      static_cast<double>(drive_torque), static_cast<double>(brake_torque),
      static_cast<double>(drive_torque), static_cast<double>(brake_torque),
      static_cast<double>(drive_torque), static_cast<double>(brake_torque));
  }
  if (valid) valid = append_literal("]}] }\n", 6);
  if (!valid || std::fwrite(request.data(), 1, offset, child_input_) != offset ||
    std::fflush(child_input_) != 0)
  {
    throw std::runtime_error("failed writing a physics tick to PhysX Odom");
  }
  ++physics_step_index_;
  physics_step_index_ += kPhysicsStepsPerPacket - 1;
  return read_state();
}

PredictedPlantState PhysxOdomRuntime::read_state()
{
  char * line = nullptr;
  size_t capacity = 0;
  const ssize_t count = getline(&line, &capacity, child_output_);
  if (count <= 0) {
    std::free(line);
    throw std::runtime_error("PhysX Odom process ended before returning a state");
  }
  const std::string response(line, static_cast<size_t>(count));
  std::free(line);

  const auto value_for = [&response](const char * key) {
      const std::string marker = std::string("\"") + key + "\":";
      const auto position = response.find(marker);
      if (position == std::string::npos) {
        throw std::runtime_error(
                std::string("PhysX Odom response lacks ") + key);
      }
      const char * begin = response.c_str() + position + marker.size();
      char * end = nullptr;
      const double value = std::strtod(begin, &end);
      if (end == begin || !std::isfinite(value)) {
        throw std::runtime_error(
                std::string("PhysX Odom returned invalid ") + key);
      }
      return value;
    };
  const auto array_for = [&response](const char * key, double * output,
                                     size_t expected_size) {
      const std::string marker = std::string("\"") + key + "\":[";
      const auto position = response.find(marker);
      if (position == std::string::npos) {
        throw std::runtime_error(std::string("PhysX Odom response lacks ") + key);
      }
      const char * cursor = response.c_str() + position + marker.size();
      for (size_t index = 0; index < expected_size; ++index) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n') ++cursor;
        char * end = nullptr;
        output[index] = std::strtod(cursor, &end);
        if (end == cursor || !std::isfinite(output[index])) {
          throw std::runtime_error(
                  std::string("PhysX Odom returned invalid array ") + key);
        }
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n') ++cursor;
        if (index + 1 < expected_size) {
          if (*cursor != ',')
            throw std::runtime_error(
                    std::string("PhysX Odom returned short array ") + key);
          ++cursor;
        }
      }
      if (*cursor != ']')
        throw std::runtime_error(
                std::string("PhysX Odom returned oversized array ") + key);
    };

  PredictedPlantState plant;
  std::array<double, 3> rearAxleWorld{};
  array_for("rear_axle_position_world_m", rearAxleWorld.data(),
            rearAxleWorld.size());
  // Unity world +Z is vehicle-forward and world -X is vehicle-left.
  plant.rear_axle_x_m = rearAxleWorld[2];
  plant.rear_axle_y_m = -rearAxleWorld[0];
  array_for("unity_source_state", plant.state.data(), plant.state.size());
  array_for("wheel_rotation_angle_rad",
            plant.wheel_rotation_angle_rad.data(),
            plant.wheel_rotation_angle_rad.size());
  array_for("wheel_encoder_angle_prediction_rad",
            plant.wheel_encoder_angle_prediction_rad.data(),
            plant.wheel_encoder_angle_prediction_rad.size());
  array_for("longitudinal_slip", plant.longitudinal_slip.data(), 4);
  array_for("lateral_slip", plant.lateral_slip.data(), 4);
  array_for("normal_load_n", plant.normal_load_n.data(), 4);
  array_for("suspension_force_n", plant.suspension_force_n.data(), 4);
  array_for("suspension_jounce_rate_mps",
            plant.suspension_jounce_rate_mps.data(), 4);
  array_for("tire_longitudinal_force_n",
            plant.tire_longitudinal_force_n.data(), 4);
  array_for("tire_lateral_force_n", plant.tire_lateral_force_n.data(), 4);
  plant.longitudinal_acceleration_mps2 =
      value_for("longitudinal_acceleration_mps2");
  plant.lateral_acceleration_mps2 = value_for("lateral_acceleration_mps2");
  plant.yaw_acceleration_radps2 = value_for("yaw_acceleration_radps2");
  return plant;
}

}  // namespace f1tenth_localization

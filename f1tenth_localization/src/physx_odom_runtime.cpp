#include "f1tenth_localization/physx_odom_runtime.hpp"

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

constexpr unsigned int kPhysicsStepsPerPacket = 25;
constexpr unsigned int kInitialRestSettleSteps = 1000;
constexpr float kPhysicsStepS = 0.001f;
constexpr float kSteeringRateDegPerS = 183.346f;
constexpr float kSteeringLimitDeg = 30.0f;
constexpr float kWheelbaseMm = 324.0f;
constexpr float kTrackWidthMm = 236.0f;
constexpr float kMotorTorqueNm = 428.0f;
constexpr float kPi = 3.14159265358979323846f;

float unity_steering_step(float angle_deg, float command_norm)
{
  const float command = std::clamp(command_norm, -1.0f, 1.0f);
  const float threshold = angle_deg / kSteeringLimitDeg;
  const float increment = kSteeringRateDegPerS * kPhysicsStepS;
  const float target = -command * kSteeringLimitDeg;
  if (command > threshold) {
    angle_deg -= increment;
    if (angle_deg <= target) {
      angle_deg = target;
    }
  } else {
    angle_deg += increment;
    if (angle_deg >= target) {
      angle_deg = target;
    }
  }
  return angle_deg;
}

float radians(float degrees)
{
  return degrees * (kPi / 180.0f);
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
  for (unsigned int index = 0; index < kInitialRestSettleSteps; ++index) {
    send_physics_step(0.0f);
    (void)read_state();
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
  bool actuator_feedback_valid, double steering_feedback_rad,
  double throttle_feedback)
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
  const bool use_feedback = actuator_feedback_valid &&
    std::isfinite(steering_feedback_rad) && std::isfinite(throttle_feedback);
  const float throttle = use_feedback ? static_cast<float>(
    std::clamp(throttle_feedback, -1.0, 1.0)) : command_throttle;
  const float steering_start_deg = center_steering_angle_deg_;
  const float steering_feedback_deg = use_feedback ? std::clamp(
    -static_cast<float>(steering_feedback_rad) * (180.0f / kPi),
    -kSteeringLimitDeg, kSteeringLimitDeg) : steering_start_deg;

  PredictedPlantState plant;
  for (unsigned int index = 0; index < kPhysicsStepsPerPacket; ++index) {
    if (use_feedback) {
      const float fraction = static_cast<float>(index + 1) /
        static_cast<float>(kPhysicsStepsPerPacket);
      center_steering_angle_deg_ = steering_start_deg + fraction *
        (steering_feedback_deg - steering_start_deg);
    } else {
      center_steering_angle_deg_ = unity_steering_step(
        center_steering_angle_deg_, steering_command);
    }
    send_physics_step(throttle);
    plant = read_state();
  }
  return plant;
}

void PhysxOdomRuntime::send_physics_step(float throttle_input)
{
  if (child_input_ == nullptr) {
    throw std::runtime_error("PhysX Odom process is not available");
  }
  const auto [front_left, front_right] = unity_ackermann(center_steering_angle_deg_);
  const float drive_torque = throttle_input == 0.0f ?
    0.0f : kMotorTorqueNm * throttle_input / 4.0f;
  const float brake_torque = throttle_input == 0.0f ? kMotorTorqueNm : 0.0f;
  const double time_s = static_cast<double>(physics_step_index_) * 0.001;
  const int written = std::fprintf(
    child_input_,
    "{\"op\":\"step\",\"capture_row\":%llu,\"time_s\":%.9f,"
    "\"wheel_commands\":["
    "{\"steer_angle_rad\":%.9g,\"drive_torque_nm\":%.9g,\"brake_torque_nm\":%.9g},"
    "{\"steer_angle_rad\":%.9g,\"drive_torque_nm\":%.9g,\"brake_torque_nm\":%.9g},"
    "{\"steer_angle_rad\":0,\"drive_torque_nm\":%.9g,\"brake_torque_nm\":%.9g},"
    "{\"steer_angle_rad\":0,\"drive_torque_nm\":%.9g,\"brake_torque_nm\":%.9g}]}\n",
    static_cast<unsigned long long>(physics_step_index_), time_s,
    static_cast<double>(front_left), static_cast<double>(drive_torque),
    static_cast<double>(brake_torque), static_cast<double>(front_right),
    static_cast<double>(drive_torque), static_cast<double>(brake_torque),
    static_cast<double>(drive_torque), static_cast<double>(brake_torque),
    static_cast<double>(drive_torque), static_cast<double>(brake_torque));
  if (written < 0 || std::fflush(child_input_) != 0) {
    throw std::runtime_error("failed writing a physics tick to PhysX Odom");
  }
  ++physics_step_index_;
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
  array_for("unity_source_state", plant.state.data(), plant.state.size());
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

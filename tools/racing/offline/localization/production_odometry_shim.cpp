// C ABI adapter around the production observer and packet assembler. The
// offline loop sends synchronized synthetic sensor samples at source time;
// estimator equations and packet-completion behavior come from production.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "f1tenth_localization/odometry_observer.hpp"
#include "f1tenth_localization/sensor_packet_assembler.hpp"

namespace
{

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

template<typename T>
T parameter(const YAML::Node & params, const char * name)
{
  const auto value = params[name];
  if (!value) {
    throw std::runtime_error(std::string("missing odometry YAML parameter: ") + name);
  }
  return value.as<T>();
}

f1tenth_localization::OdometryObserverConfig load_config(
  const std::string & path, double & orientation_gain,
  double & maximum_orientation_step)
{
  const auto params = YAML::LoadFile(path)["sensor_odometry"]["ros__parameters"];
  if (!params) {
    throw std::runtime_error("sensor_odometry.ros__parameters not found");
  }
  auto config = f1tenth_localization::deployment_observer_config();
#define READ_DOUBLE(field) config.field = parameter<double>(params, #field)
#define READ_BOOL(field) config.field = parameter<bool>(params, #field)
  READ_DOUBLE(wheel_radius_m);
  config.wheel_speed_scale = std::max(
    0.0, parameter<double>(params, "wheel_speed_scale"));
  config.wheel_speed_scale_speeds_mps = parameter<std::vector<double>>(
    params, "wheel_speed_scale_speeds_mps");
  config.wheel_speed_scale_values = parameter<std::vector<double>>(
    params, "wheel_speed_scale_values");
  READ_DOUBLE(reset_encoder_jump_rad);
  config.wheel_speed_window_s = std::max(
    0.0, parameter<double>(params, "wheel_speed_window_s"));
  READ_DOUBLE(normal_packet_dt_max_s);
  READ_DOUBLE(decel_detect_ax_mps2);
  READ_DOUBLE(decel_ax_scale);
  READ_DOUBLE(decel_ax_offset_mps2);
  config.wheel_dropout_positive_ax_max_mps2 = std::max(
    0.0, parameter<double>(params, "wheel_dropout_positive_ax_max_mps2"));
  READ_DOUBLE(wheel_update_ax_abs_max_mps2);
  READ_DOUBLE(wheel_freeze_speed_mps);
  config.wheel_recovery_launch_speed_mps = std::max(
    0.0, parameter<double>(params, "wheel_recovery_launch_speed_mps"));
  config.wheel_recovery_launch_innovation_mps = std::max(
    0.0, parameter<double>(params, "wheel_recovery_launch_innovation_mps"));
  config.wheel_recovery_launch_wheel_speed_mps = std::max(
    0.0, parameter<double>(params, "wheel_recovery_launch_wheel_speed_mps"));
  READ_DOUBLE(wheel_burst_disagreement_mps);
  config.wheel_burst_catchup_accel_mps2 = std::max(
    0.0, parameter<double>(params, "wheel_burst_catchup_accel_mps2"));
  config.wheel_burst_catchup_max_mps = std::max(
    0.0, parameter<double>(params, "wheel_burst_catchup_max_mps"));
  READ_BOOL(allow_turn_current_packet_recovery);
  config.turn_current_packet_max_increase_mps = std::max(
    0.0, parameter<double>(params, "turn_current_packet_max_increase_mps"));
  config.turn_current_packet_max_decrease_mps = std::max(
    0.0, parameter<double>(params, "turn_current_packet_max_decrease_mps"));
  READ_BOOL(use_turn_speed_bias_model);
  READ_DOUBLE(turn_speed_bias_constant_mps);
  READ_DOUBLE(turn_speed_bias_speed_mps);
  READ_DOUBLE(turn_speed_bias_speed_squared_mps);
  READ_DOUBLE(turn_speed_bias_yaw_rate_abs_mps);
  READ_DOUBLE(turn_speed_bias_yaw_rate_squared_mps);
  READ_DOUBLE(turn_speed_bias_speed_yaw_rate_abs_mps);
  config.turn_speed_bias_max_mps = std::max(
    0.0, parameter<double>(params, "turn_speed_bias_max_mps"));
  READ_BOOL(use_coherent_packet_velocity_for_pose);
  config.coherent_packet_pose_blend = std::clamp(
    parameter<double>(params, "coherent_packet_pose_blend"), 0.0, 1.0);
  config.wheel_speed_slew_limit_mps2 = std::max(
    0.0, parameter<double>(params, "wheel_speed_slew_limit_mps2"));
  READ_DOUBLE(wheel_update_beta);
  READ_DOUBLE(stationary_speed_threshold_mps);
  READ_DOUBLE(stationary_hold_s);
  READ_DOUBLE(stationary_ax_abs_max_mps2);
  READ_DOUBLE(stationary_ay_abs_max_mps2);
  READ_DOUBLE(stationary_yaw_rate_abs_max_radps);
  READ_DOUBLE(turn_enter_yaw_rate_radps);
  READ_DOUBLE(turn_enter_abs_ay_mps2);
  READ_DOUBLE(turn_exit_yaw_rate_radps);
  READ_DOUBLE(turn_exit_abs_ay_mps2);
  READ_DOUBLE(turn_exit_hold_s);
  READ_DOUBLE(turn_wheel_braking_ax_mps2);
  READ_BOOL(integrate_lateral_acceleration_in_turn);
  READ_BOOL(use_kinematic_lateral_slip_model);
  READ_DOUBLE(lateral_velocity_yaw_rate_gain_m);
  READ_DOUBLE(lateral_velocity_speed_yaw_rate_gain_s);
  config.lateral_velocity_max_mps = std::max(
    0.0, parameter<double>(params, "lateral_velocity_max_mps"));
  READ_DOUBLE(lateral_velocity_reference_forward_offset_m);
  READ_DOUBLE(imu_acceleration_reference_x_m);
  READ_DOUBLE(max_imu_ax_abs_mps2);
  orientation_gain = parameter<double>(params, "imu_orientation_correction_gain");
  maximum_orientation_step = parameter<double>(params, "max_imu_orientation_step_rad");
#undef READ_DOUBLE
#undef READ_BOOL
  return config;
}

class OfflineProductionOdometry
{
public:
  explicit OfflineProductionOdometry(const std::string & yaml_path)
  : config_(load_config(yaml_path, orientation_gain_, max_orientation_step_)),
    observer_(config_)
  {
    assembler_.set_packet_callback(
      [this](const f1tenth_localization::SensorPacket & packet) {
        process(packet);
      });
  }

  void reset()
  {
    assembler_.reset();
    observer_.reset();
    yaw_initialized_ = false;
    yaw_reference_rad_ = 0.0;
    previous_raw_relative_yaw_rad_ = 0.0;
    continuous_yaw_rad_ = 0.0;
    previous_yaw_stamp_s_ = 0.0;
    previous_yaw_rate_radps_ = 0.0;
    has_output_ = false;
  }

  void add_left(std::int64_t stamp_ns, double angle)
  {
    assembler_.add_encoder_sample(stamp_ns, angle, true);
  }

  void add_right(std::int64_t stamp_ns, double angle)
  {
    assembler_.add_encoder_sample(stamp_ns, angle, false);
  }

  void add_imu(std::int64_t stamp_ns, double ax, double ay,
               double yaw_rate, double yaw)
  {
    assembler_.add_imu_sample(stamp_ns, ax, ay, yaw_rate, yaw);
  }

  bool output(double * values, std::size_t count) const
  {
    if (!has_output_ || count < 23) {
      return false;
    }
    const auto & e = last_;
    const double fields[23] = {
      e.stamp_s, e.dt_s, e.speed_pred_mps, e.speed_mps,
      e.body_u_mps, e.body_v_mps, e.x_m, e.y_m, e.yaw_rad,
      e.wheel_raw_mps, e.wheel_mapped_mps, e.wheel_packet_mps,
      e.turn_speed_bias_mps, e.ax_mps2, e.ay_mps2, e.yaw_rate_radps,
      e.wheel_update_used ? 1.0 : 0.0,
      e.wheel_burst_rejected ? 1.0 : 0.0,
      e.turn_mode ? 1.0 : 0.0, e.reset_epoch ? 1.0 : 0.0,
      e.timing_degraded ? 1.0 : 0.0,
      e.sensor_outlier ? 1.0 : 0.0, e.valid ? 1.0 : 0.0,
    };
    std::copy(fields, fields + 23, values);
    return true;
  }

private:
  double continuous_yaw(const f1tenth_localization::SensorPacket & packet)
  {
    const double raw_yaw = packet.imu_yaw_rad;
    if (!yaw_initialized_) {
      yaw_initialized_ = true;
      yaw_reference_rad_ = raw_yaw;
      previous_raw_relative_yaw_rad_ = 0.0;
      continuous_yaw_rad_ = 0.0;
      previous_yaw_stamp_s_ = static_cast<double>(packet.stamp_ns) * 1.0e-9;
      previous_yaw_rate_radps_ = packet.yaw_rate_radps;
      return continuous_yaw_rad_;
    }
    const double stamp_s = static_cast<double>(packet.stamp_ns) * 1.0e-9;
    const double dt = stamp_s - previous_yaw_stamp_s_;
    const double raw_relative = wrap_angle(raw_yaw - yaw_reference_rad_);
    const double raw_delta = wrap_angle(raw_relative - previous_raw_relative_yaw_rad_);
    if (dt > 0.0 && dt <= 0.5) {
      continuous_yaw_rad_ = wrap_angle(
        continuous_yaw_rad_ + packet.yaw_rate_radps * dt);
      if (std::abs(raw_delta) <= max_orientation_step_) {
        continuous_yaw_rad_ = wrap_angle(
          continuous_yaw_rad_ + orientation_gain_ *
          wrap_angle(raw_relative - continuous_yaw_rad_));
      } else {
        yaw_reference_rad_ = wrap_angle(raw_yaw - continuous_yaw_rad_);
        previous_raw_relative_yaw_rad_ = continuous_yaw_rad_;
      }
    }
    if (std::abs(raw_delta) <= max_orientation_step_) {
      previous_raw_relative_yaw_rad_ = raw_relative;
    }
    previous_yaw_stamp_s_ = stamp_s;
    previous_yaw_rate_radps_ = packet.yaw_rate_radps;
    return continuous_yaw_rad_;
  }

  void process(const f1tenth_localization::SensorPacket & packet)
  {
    f1tenth_localization::OdometryObservation observation;
    observation.stamp_s = static_cast<double>(packet.stamp_ns) * 1.0e-9;
    observation.left_angle_rad = packet.left_angle_rad;
    observation.right_angle_rad = packet.right_angle_rad;
    observation.ax_mps2 = packet.ax_mps2;
    observation.ay_mps2 = packet.ay_mps2;
    observation.yaw_rate_radps = packet.yaw_rate_radps;
    observation.yaw_rad = continuous_yaw(packet);
    last_ = observer_.update(observation);
    has_output_ = true;
  }

  f1tenth_localization::OdometryObserverConfig config_;
  f1tenth_localization::OdometryObserver observer_;
  f1tenth_localization::SensorPacketAssembler assembler_;
  double orientation_gain_{1.0};
  double max_orientation_step_{0.30};
  bool yaw_initialized_{false};
  double yaw_reference_rad_{0.0};
  double previous_raw_relative_yaw_rad_{0.0};
  double continuous_yaw_rad_{0.0};
  double previous_yaw_stamp_s_{0.0};
  double previous_yaw_rate_radps_{0.0};
  bool has_output_{false};
  f1tenth_localization::OdometryEstimate last_{};
};

}  // namespace

extern "C"
{

void * offline_odom_create(const char * yaml_path, char * error,
                           std::size_t error_capacity)
{
  try {
    return new OfflineProductionOdometry(yaml_path);
  } catch (const std::exception & exception) {
    if (error && error_capacity > 0) {
      const auto length = std::min(error_capacity - 1,
                                   std::char_traits<char>::length(exception.what()));
      std::copy(exception.what(), exception.what() + length, error);
      error[length] = '\0';
    }
    return nullptr;
  }
}

void offline_odom_destroy(void * handle)
{
  delete static_cast<OfflineProductionOdometry *>(handle);
}

void offline_odom_reset(void * handle)
{
  static_cast<OfflineProductionOdometry *>(handle)->reset();
}

void offline_odom_add_left(void * handle, std::int64_t stamp_ns, double angle)
{
  static_cast<OfflineProductionOdometry *>(handle)->add_left(stamp_ns, angle);
}

void offline_odom_add_right(void * handle, std::int64_t stamp_ns, double angle)
{
  static_cast<OfflineProductionOdometry *>(handle)->add_right(stamp_ns, angle);
}

void offline_odom_add_imu(void * handle, std::int64_t stamp_ns,
                          double ax, double ay, double yaw_rate, double yaw)
{
  static_cast<OfflineProductionOdometry *>(handle)->add_imu(
    stamp_ns, ax, ay, yaw_rate, yaw);
}

int offline_odom_get_output(void * handle, double * values, std::size_t count)
{
  return static_cast<OfflineProductionOdometry *>(handle)->output(values, count)
    ? 1 : 0;
}

}

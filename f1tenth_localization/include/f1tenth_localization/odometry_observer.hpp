#pragma once

#include <cstdint>

namespace f1tenth_localization
{

struct OdometryObserverConfig
{
  double wheel_radius_m{0.059};
  double reset_encoder_jump_rad{50.0};
  double normal_packet_dt_max_s{0.040};
  double degraded_packet_dt_max_s{0.100};
  double decel_detect_ax_mps2{-0.5};
  double decel_ax_scale{1.005};
  double decel_ax_offset_mps2{0.020};
  double wheel_update_ax_abs_max_mps2{0.6};
  double wheel_freeze_speed_mps{0.15};
  double wheel_innovation_max_mps{0.30};
  double wheel_update_beta{0.20};
  double turn_enter_yaw_rate_radps{0.6};
  double turn_enter_abs_ay_mps2{6.0};
  double turn_exit_yaw_rate_radps{0.1};
  double turn_exit_abs_ay_mps2{0.5};
  double turn_exit_hold_s{0.5};
  double imu_x_offset_m{0.08};
};

struct OdometryObservation
{
  double stamp_s{0.0};
  double left_angle_rad{0.0};
  double right_angle_rad{0.0};
  double ax_mps2{0.0};
  double ay_mps2{0.0};
  double yaw_rate_radps{0.0};
  double yaw_rad{0.0};
};

struct OdometryEstimate
{
  double stamp_s{0.0};
  double dt_s{0.0};
  double speed_pred_mps{0.0};
  double speed_mps{0.0};
  double body_u_mps{0.0};
  double body_v_mps{0.0};
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
  double wheel_raw_mps{0.0};
  double wheel_mapped_mps{0.0};
  double ax_mps2{0.0};
  double ay_mps2{0.0};
  double yaw_rate_radps{0.0};
  bool wheel_update_used{false};
  bool turn_mode{false};
  bool reset_epoch{false};
  bool timing_degraded{false};
  bool valid{false};
};

class OdometryObserver final
{
public:
  explicit OdometryObserver(OdometryObserverConfig config = {});

  void reset() noexcept;

  OdometryEstimate update(const OdometryObservation & observation) noexcept;

private:
  static double wrap_angle(double angle) noexcept;
  static bool finite(double value) noexcept;

  OdometryEstimate estimate(const OdometryObservation & observation) const noexcept;
  void update_pose(double dt_s, double yaw_rad) noexcept;
  void update_turn(double ax_origin, double ay_origin, double yaw_rate, double dt_s) noexcept;

  OdometryObserverConfig config_;
  bool initialized_{false};
  bool turn_mode_{false};
  double turn_calm_time_s_{0.0};
  double previous_stamp_s_{0.0};
  double previous_left_angle_rad_{0.0};
  double previous_right_angle_rad_{0.0};
  double previous_yaw_rad_{0.0};
  double previous_yaw_rate_radps_{0.0};
  double speed_mps_{0.0};
  double body_u_mps_{0.0};
  double body_v_mps_{0.0};
  double x_m_{0.0};
  double y_m_{0.0};
  double last_speed_pred_mps_{0.0};
  double last_wheel_raw_mps_{0.0};
  double last_wheel_mapped_mps_{0.0};
};

}  // namespace f1tenth_localization

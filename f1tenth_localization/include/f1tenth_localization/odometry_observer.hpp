#pragma once

#include <cstdint>

namespace f1tenth_localization
{

struct OdometryObserverConfig
{
  double wheel_radius_m{0.059};
  double reset_encoder_jump_rad{50.0};
  // The competition player currently publishes native telemetry at about
  // 20 Hz (roughly 0.050 s). Keep that cadence in the normal, non-degraded
  // path; a 40 ms threshold falsely marked every track packet degraded and
  // inflated /odom covariance to the controller stop threshold.
  double normal_packet_dt_max_s{0.080};
  double degraded_packet_dt_max_s{0.100};
  // Short gaps still have valid encoder endpoints and can be integrated as an
  // average displacement. Longer gaps are rebaselined conservatively.
  double max_integratable_gap_s{0.250};
  double decel_detect_ax_mps2{-0.5};
  double decel_ax_scale{1.005};
  double decel_ax_offset_mps2{0.020};
  double wheel_update_ax_abs_max_mps2{0.6};
  double wheel_freeze_speed_mps{0.15};
  double wheel_innovation_max_mps{0.30};
  double wheel_update_beta{0.20};
  // A frozen encoder must not immediately zero a moving estimate because a
  // single dropped packet is possible.  Sustained zero wheel motion together
  // with calm IMU data is, however, a reliable stopped/collision signature.
  // This is deliberately much lower than wheel_freeze_speed_mps: the latter
  // is an innovation/missing-packet guard, not a stopped-vehicle threshold.
  double stationary_speed_threshold_mps{0.03};
  double stationary_hold_s{0.10};
  double stationary_ax_abs_max_mps2{0.25};
  double stationary_ay_abs_max_mps2{0.75};
  double stationary_yaw_rate_abs_max_radps{0.15};
  double turn_enter_yaw_rate_radps{0.6};
  double turn_enter_abs_ay_mps2{6.0};
  double turn_exit_yaw_rate_radps{0.1};
  double turn_exit_abs_ay_mps2{0.5};
  double turn_exit_hold_s{0.5};
  double imu_x_offset_m{0.08};
  // The simulator's lateral IMU acceleration contains enough bias/noise to
  // create a persistent pose error when integrated at native 20 Hz.  The
  // default car model therefore uses wheel speed plus IMU yaw only (a
  // no-lateral-slip kinematic update). Keep the dynamic option available for
  // offline comparison and vehicles with a separately validated slip model.
  bool integrate_lateral_acceleration_in_turn{false};
  // A single impossible longitudinal IMU sample must not be integrated into
  // odometry.  The competition vehicle's lateral acceleration can be large
  // in a tight turn, so this guard is intentionally only for ax.
  double max_imu_ax_abs_mps2{30.0};
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
  // Exact synchronized packet inputs used by the observer. These are kept in
  // diagnostics so an offline replay cannot accidentally combine callback
  // snapshots from different source timestamps.
  double left_angle_rad{0.0};
  double right_angle_rad{0.0};
  double imu_yaw_rad{0.0};
  double wheel_raw_mps{0.0};
  double wheel_mapped_mps{0.0};
  double ax_mps2{0.0};
  double ay_mps2{0.0};
  double yaw_rate_radps{0.0};
  bool wheel_update_used{false};
  bool turn_mode{false};
  bool reset_epoch{false};
  bool timing_degraded{false};
  bool sensor_outlier{false};
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
  double stationary_time_s_{0.0};
};

}  // namespace f1tenth_localization

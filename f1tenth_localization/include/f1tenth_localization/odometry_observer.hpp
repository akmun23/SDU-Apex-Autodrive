#pragma once

#include <cstdint>
#include <deque>

namespace f1tenth_localization
{

struct OdometryObserverConfig
{
  double wheel_radius_m{0.059};
  // Direct wheel-to-body speed scale fit from the clean single-simulator
  // 4 m/s track run. Keep the physical wheel radius separate from this
  // runtime calibration.
  double wheel_speed_scale{0.968};
  double reset_encoder_jump_rad{50.0};
  // The simulator can repeat a cumulative encoder angle for several source
  // packets and then publish the accumulated jump.  Differentiate over a
  // short source-time window so the observer sees displacement, not bursts.
  // The 20 Hz simulator encoder stream contains delayed cumulative-angle
  // packets.  The recorded run comparison showed that 150 ms adds avoidable
  // launch/braking lag; retain only two native samples for the rolling rate.
  double wheel_speed_window_s{0.10};
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
  // Wheel speed is the direct longitudinal measurement.  Keep this gate
  // above the measured launch acceleration so valid wheel updates are not
  // replaced by the slower IMU prediction.
  double wheel_update_ax_abs_max_mps2{6.5};
  double wheel_freeze_speed_mps{0.15};
  // Permit recovery from a transient IMU-speed error while retaining the
  // frozen-wheel and timing gates for missing packets.
  double wheel_innovation_max_mps{1.50};
  // A delayed cumulative-encoder burst appears first in the rolling rate and
  // then again in the current packet. Reject only that two-stage signature;
  // ordinary acceleration packets have no packet-vs-window disagreement.
  double wheel_burst_disagreement_mps{1.0};
  double wheel_update_beta{0.85};
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
  // During a hard turn/braking transient the IMU-integrated prediction can
  // temporarily exceed the synchronized wheel estimate by more than the
  // normal innovation gate.  A valid nonzero wheel packet is still useful in
  // that case; isolated zero packets remain protected by wheel_freeze_speed.
  double turn_wheel_braking_ax_mps2{-1.0};
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
  // Diagnostic-only current-packet rate. The observer normally uses the
  // short window rate because the simulator can burst cumulative angle.
  double wheel_packet_mps{0.0};
  double ax_mps2{0.0};
  double ay_mps2{0.0};
  double yaw_rate_radps{0.0};
  bool wheel_update_used{false};
  bool wheel_burst_rejected{false};
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
  void update_pose(
    double dt_s, double yaw_rad,
    double previous_body_u_mps, double previous_body_v_mps) noexcept;
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
  double last_wheel_packet_mps_{0.0};
  double stationary_time_s_{0.0};
  bool wheel_dropout_active_{false};
  bool wheel_burst_rejected_{false};
  // Keep a burst dropout active until the current packet also agrees with
  // the rolling-window rate; the first nonzero packet can still be delayed.
  bool wheel_burst_recovery_pending_{false};

  struct EncoderSample
  {
    double stamp_s;
    double left_angle_rad;
    double right_angle_rad;
  };
  std::deque<EncoderSample> encoder_history_;
};

}  // namespace f1tenth_localization

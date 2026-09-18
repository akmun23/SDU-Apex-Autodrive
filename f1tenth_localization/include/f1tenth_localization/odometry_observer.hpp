#pragma once

#include <cstdint>
#include <deque>
#include <vector>

namespace f1tenth_localization
{

struct OdometryObserverConfig
{
  double wheel_radius_m{0.059};
  // Fallback direct wheel-to-body speed scale. Keep the physical wheel radius
  // separate from this runtime calibration.
  double wheel_speed_scale{0.968};
  // Optional monotone calibration of the wheel-to-body speed scale. The
  // values are linearly interpolated against the raw rolling wheel speed and
  // endpoint values are held outside the identified range. An invalid or
  // incomplete table falls back to wheel_speed_scale, so a bad deployment
  // parameter cannot silently disable odometry.
  std::vector<double> wheel_speed_scale_speeds_mps;
  std::vector<double> wheel_speed_scale_values;
  double reset_encoder_jump_rad{50.0};
  // The simulator can repeat a cumulative encoder angle for several source
  // packets and then publish the accumulated jump.  Differentiate over a
  // short source-time window so the observer sees displacement, not bursts.
  // The 20 Hz simulator encoder stream contains delayed cumulative-angle
  // packets.  The recorded run comparison showed that 150 ms adds avoidable
  // launch/braking lag; retain only two native samples for the rolling rate.
  double wheel_speed_window_s{0.10};
  // This nominal interval affects diagnostics only; delayed samples are kept.
  double normal_packet_dt_max_s{0.035};
  double decel_detect_ax_mps2{-0.5};
  // The repeated-encoder dropout can coincide with genuine hard braking. Use
  // the calibrated IMU deceleration until a fresh wheel packet returns; a
  // reduced scale leaves /odom dangerously above the actual vehicle speed in
  // the closed loop.
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
  // During launch the encoder can report a coherent wheel rate while the
  // driven wheels are spinning far faster than the body. Do not let the
  // coherent-recovery shortcut promote that value into odometry until the
  // body-speed prediction has left the launch regime.
  double wheel_recovery_launch_speed_mps{2.0};
  double wheel_recovery_launch_innovation_mps{2.0};
  double wheel_recovery_launch_wheel_speed_mps{4.0};
  // A delayed cumulative-encoder burst appears first in the rolling rate and
  // then again in the current packet. Reject only that two-stage signature;
  // ordinary acceleration packets have no packet-vs-window disagreement.
  double wheel_burst_disagreement_mps{1.0};
  // In a turn, a fresh synchronized packet can be a valid current-motion
  // sample while the rolling window still contains a repeated angle. Permit
  // that recovery only when the packet is close to the causal speed and is
  // clearly newer than the stale window; straight-line burst recovery keeps
  // the stricter window-coherence rule.
  bool allow_turn_current_packet_recovery{true};
  double turn_current_packet_max_increase_mps{0.20};
  // Provisional, bounded turn-speed residual calibration identified from
  // held-out live runs. It uses only mapped wheel speed and IMU yaw rate at
  // runtime; simulator truth is used offline to fit/score the coefficients.
  bool use_turn_speed_bias_model{false};
  double turn_speed_bias_constant_mps{0.0};
  double turn_speed_bias_speed_mps{0.0};
  double turn_speed_bias_speed_squared_mps{0.0};
  double turn_speed_bias_yaw_rate_abs_mps{0.0};
  double turn_speed_bias_yaw_rate_squared_mps{0.0};
  double turn_speed_bias_speed_yaw_rate_abs_mps{0.0};
  double turn_speed_bias_max_mps{0.10};
  // The rolling encoder window rejects delayed cumulative-angle bursts, but
  // it lags the current motion during a genuine turn transient. When both
  // rates are coherent, use the current packet only for pose integration;
  // keep the published longitudinal state anchored to the rolling estimate.
  bool use_coherent_packet_velocity_for_pose{false};
  double coherent_packet_pose_blend{1.0};
  // Reject abrupt rolling wheel-rate changes that are physically inconsistent
  // with the causal body-speed estimate. A release back to causal speed is
  // still allowed to recover.
  double wheel_speed_slew_limit_mps2{40.0};
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
  // Longitudinal location of the point whose velocity is differentiated by
  // the Unity IMU script, relative to base_link/rear axle. This is not the
  // IMU Transform offset used for TF (imu_x_m): Unity differentiates the
  // vehicle Rigidbody velocity at its configured COM.
  double imu_acceleration_reference_x_m{0.15532};
  // The simulator's lateral IMU acceleration contains enough bias/noise to
  // create a persistent pose error when integrated at native 20 Hz.  The
  // default car model therefore uses wheel speed plus IMU yaw only (a
  // no-lateral-slip kinematic update). The implementation automatically
  // enables the dynamic option during an explicitly rejected wheel-slip or
  // dropout interval; this flag remains available for offline comparison and
  // vehicles with a separately validated slip model.
  bool integrate_lateral_acceleration_in_turn{false};
  // Optional bounded COM lateral-velocity model using causal wheel speed and
  // yaw rate. Its output is translated to base_link with the configured point
  // offset; it is separate from integrating lateral acceleration.
  bool use_kinematic_lateral_slip_model{false};
  // COM model: v_com = yaw_rate * (gain_m + gain_s * u), before its bound.
  double lateral_velocity_yaw_rate_gain_m{0.167};
  double lateral_velocity_speed_yaw_rate_gain_s{-0.0063};
  // Bound applied to the estimated COM lateral velocity, before point shift.
  double lateral_velocity_max_mps{0.35};
  // The kinematic lateral-velocity model estimates velocity at the Rigidbody
  // COM. Translate it to the odometry base point, which is behind the COM:
  // v_base = v_reference - yaw_rate * reference_forward_offset.
  double lateral_velocity_reference_forward_offset_m{0.0};
  // A single impossible longitudinal IMU sample must not be integrated into
  // odometry.  The competition vehicle's lateral acceleration can be large
  // in a tight turn, so this guard is intentionally only for ax.
  double max_imu_ax_abs_mps2{30.0};
};

/* Configuration used by offline replay and as the node's pre-YAML baseline.
 * Keep this profile aligned with config/sensor_odometry.yaml so a replay
 * cannot silently evaluate a different estimator than the deployed node. */
OdometryObserverConfig deployment_observer_config();

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
  double turn_speed_bias_mps{0.0};
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
  double wheel_scale_for_speed(double raw_speed_mps) const noexcept;
  double turn_speed_bias_mps(
    double wheel_mapped_mps, double yaw_rate_radps) const noexcept;
  double kinematic_base_lateral_velocity(
    double yaw_rate_radps, double longitudinal_speed_mps) const noexcept;

  OdometryEstimate estimate(const OdometryObservation & observation) const noexcept;
  void update_pose(
    double dt_s, double yaw_rad,
    double previous_body_u_mps, double previous_body_v_mps,
    double body_u_mps, double body_v_mps) noexcept;
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
  double previous_pose_body_u_mps_{0.0};
  double previous_pose_body_v_mps_{0.0};
  double x_m_{0.0};
  double y_m_{0.0};
  double last_speed_pred_mps_{0.0};
  double last_wheel_raw_mps_{0.0};
  double last_wheel_mapped_mps_{0.0};
  double last_wheel_packet_mps_{0.0};
  double last_turn_speed_bias_mps_{0.0};
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

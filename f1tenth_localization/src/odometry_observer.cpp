#include "f1tenth_localization/odometry_observer.hpp"

#include <algorithm>
#include <array>
#include <cmath>


namespace f1tenth_localization
{

OdometryObserverConfig deployment_observer_config()
{
  OdometryObserverConfig config;
  config.wheel_radius_m = 0.059;
  config.wheel_speed_scale = 0.968;
  config.wheel_speed_scale_speeds_mps = {
    0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0,
    10.0, 12.0, 14.0};
  config.wheel_speed_scale_values = {
    1.0, 0.998, 0.994, 0.992, 0.990, 0.9815,
    0.9735, 0.9658, 0.9587, 0.9520, 0.9445};
  config.reset_encoder_jump_rad = 50.0;
  config.wheel_speed_window_s = 0.10;
  config.normal_packet_dt_max_s = 0.035;
  config.degraded_packet_dt_max_s = 0.050;
  config.max_integratable_gap_s = 0.250;
  config.decel_detect_ax_mps2 = -0.5;
  config.decel_ax_scale = 1.005;
  config.decel_ax_offset_mps2 = 0.020;
  config.wheel_update_ax_abs_max_mps2 = 6.5;
  config.wheel_freeze_speed_mps = 0.15;
  config.wheel_innovation_max_mps = 1.50;
  config.wheel_recovery_launch_speed_mps = 2.0;
  config.wheel_recovery_launch_innovation_mps = 2.0;
  config.wheel_recovery_launch_wheel_speed_mps = 4.0;
  config.wheel_burst_disagreement_mps = 1.0;
  config.allow_turn_current_packet_recovery = true;
  config.turn_current_packet_max_increase_mps = 0.20;
  config.use_turn_speed_bias_model = true;
  config.turn_speed_bias_constant_mps = -0.03;
  config.turn_speed_bias_speed_mps = 0.0;
  config.turn_speed_bias_speed_squared_mps = 0.0;
  config.turn_speed_bias_yaw_rate_abs_mps = 0.0;
  config.turn_speed_bias_yaw_rate_squared_mps = 0.0;
  config.turn_speed_bias_speed_yaw_rate_abs_mps = 0.0;
  config.turn_speed_bias_max_mps = 0.03;
  config.use_coherent_packet_velocity_for_pose = true;
  config.coherent_packet_pose_blend = 1.0;
  config.wheel_speed_slew_limit_mps2 = 40.0;
  config.wheel_update_beta = 0.85;
  config.stationary_speed_threshold_mps = 0.03;
  config.stationary_hold_s = 0.10;
  config.stationary_ax_abs_max_mps2 = 0.25;
  config.stationary_ay_abs_max_mps2 = 0.75;
  config.stationary_yaw_rate_abs_max_radps = 0.15;
  config.turn_enter_yaw_rate_radps = 0.6;
  config.turn_enter_abs_ay_mps2 = 6.0;
  config.turn_exit_yaw_rate_radps = 0.1;
  config.turn_exit_abs_ay_mps2 = 0.5;
  config.turn_exit_hold_s = 0.5;
  config.turn_wheel_braking_ax_mps2 = -1.0;
  config.integrate_lateral_acceleration_in_turn = false;
  config.use_kinematic_lateral_slip_model = true;
  config.lateral_velocity_yaw_rate_gain_m = 0.167;
  config.lateral_velocity_speed_yaw_rate_gain_s = -0.0063;
  config.lateral_velocity_max_mps = 0.35;
  config.max_imu_ax_abs_mps2 = 30.0;
  config.imu_x_offset_m = 0.08;
  return config;
}

OdometryObserver::OdometryObserver(OdometryObserverConfig config)
: config_(config)
{
  reset();
}

void OdometryObserver::reset() noexcept
{
  initialized_ = false;
  turn_mode_ = false;
  turn_calm_time_s_ = 0.0;
  previous_stamp_s_ = 0.0;
  previous_left_angle_rad_ = 0.0;
  previous_right_angle_rad_ = 0.0;
  previous_yaw_rad_ = 0.0;
  previous_yaw_rate_radps_ = 0.0;
  speed_mps_ = 0.0;
  body_u_mps_ = 0.0;
  body_v_mps_ = 0.0;
  previous_pose_body_u_mps_ = 0.0;
  previous_pose_body_v_mps_ = 0.0;
  x_m_ = 0.0;
  y_m_ = 0.0;
  last_speed_pred_mps_ = 0.0;
  last_wheel_raw_mps_ = 0.0;
  last_wheel_mapped_mps_ = 0.0;
  last_wheel_packet_mps_ = 0.0;
  last_turn_speed_bias_mps_ = 0.0;
  stationary_time_s_ = 0.0;
  wheel_dropout_active_ = false;
  wheel_burst_rejected_ = false;
  wheel_burst_recovery_pending_ = false;
  encoder_history_.clear();
}

double OdometryObserver::wrap_angle(double angle) noexcept
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

bool OdometryObserver::finite(double value) noexcept
{
  return std::isfinite(value);
}

double OdometryObserver::wheel_scale_for_speed(double raw_speed_mps) const noexcept
{
  const double fallback = std::max(0.0, config_.wheel_speed_scale);
  const auto & speeds = config_.wheel_speed_scale_speeds_mps;
  const auto & values = config_.wheel_speed_scale_values;
  if (speeds.size() < 2 || speeds.size() != values.size() ||
    !finite(raw_speed_mps))
  {
    return fallback;
  }

  for (std::size_t index = 0; index < speeds.size(); ++index) {
    if (!finite(speeds[index]) || !finite(values[index]) || values[index] < 0.0 ||
      (index > 0 && speeds[index] <= speeds[index - 1]))
    {
      return fallback;
    }
  }

  const double speed = std::max(0.0, raw_speed_mps);
  if (speed <= speeds.front()) {
    return values.front();
  }
  if (speed >= speeds.back()) {
    return values.back();
  }
  const auto upper = std::upper_bound(speeds.begin(), speeds.end(), speed);
  const std::size_t index = static_cast<std::size_t>(upper - speeds.begin());
  const double lower_speed = speeds[index - 1];
  const double upper_speed = speeds[index];
  const double fraction = (speed - lower_speed) / (upper_speed - lower_speed);
  return values[index - 1] + fraction * (values[index] - values[index - 1]);
}

double OdometryObserver::turn_speed_bias_mps(
  double wheel_mapped_mps, double yaw_rate_radps) const noexcept
{
  if (!config_.use_turn_speed_bias_model || !finite(wheel_mapped_mps) ||
    !finite(yaw_rate_radps) || config_.turn_speed_bias_max_mps <= 0.0)
  {
    return 0.0;
  }
  const double speed = std::max(0.0, wheel_mapped_mps);
  const double yaw_rate_abs = std::abs(yaw_rate_radps);
  const double bias = config_.turn_speed_bias_constant_mps +
    config_.turn_speed_bias_speed_mps * speed +
    config_.turn_speed_bias_speed_squared_mps * speed * speed +
    config_.turn_speed_bias_yaw_rate_abs_mps * yaw_rate_abs +
    config_.turn_speed_bias_yaw_rate_squared_mps * yaw_rate_abs * yaw_rate_abs +
    config_.turn_speed_bias_speed_yaw_rate_abs_mps * speed * yaw_rate_abs;
  return std::clamp(
    bias, -config_.turn_speed_bias_max_mps, config_.turn_speed_bias_max_mps);
}

double OdometryObserver::kinematic_lateral_velocity(
  double yaw_rate_radps, double longitudinal_speed_mps) const noexcept
{
  if (!config_.use_kinematic_lateral_slip_model ||
    !finite(yaw_rate_radps) || !finite(longitudinal_speed_mps) ||
    !finite(config_.lateral_velocity_yaw_rate_gain_m) ||
    !finite(config_.lateral_velocity_speed_yaw_rate_gain_s) ||
    config_.lateral_velocity_max_mps <= 0.0)
  {
    return 0.0;
  }

  const double forward_speed = std::max(0.0, longitudinal_speed_mps);
  const double speed_gain = config_.lateral_velocity_yaw_rate_gain_m +
    config_.lateral_velocity_speed_yaw_rate_gain_s * forward_speed;
  return std::clamp(
    yaw_rate_radps * speed_gain,
    -config_.lateral_velocity_max_mps, config_.lateral_velocity_max_mps);
}

OdometryEstimate OdometryObserver::estimate(
  const OdometryObservation & observation) const noexcept
{
  OdometryEstimate output;
  output.stamp_s = observation.stamp_s;
  output.dt_s = initialized_ ? observation.stamp_s - previous_stamp_s_ : 0.0;
  output.speed_pred_mps = last_speed_pred_mps_;
  output.speed_mps = speed_mps_;
  output.body_u_mps = body_u_mps_;
  output.body_v_mps = body_v_mps_;
  output.x_m = x_m_;
  output.y_m = y_m_;
  output.yaw_rad = observation.yaw_rad;
  output.left_angle_rad = observation.left_angle_rad;
  output.right_angle_rad = observation.right_angle_rad;
  output.imu_yaw_rad = observation.yaw_rad;
  output.wheel_raw_mps = last_wheel_raw_mps_;
  output.wheel_mapped_mps = last_wheel_mapped_mps_;
  output.wheel_packet_mps = last_wheel_packet_mps_;
  output.turn_speed_bias_mps = last_turn_speed_bias_mps_;
  output.wheel_burst_rejected = wheel_burst_rejected_;
  output.ax_mps2 = observation.ax_mps2;
  output.ay_mps2 = observation.ay_mps2;
  output.yaw_rate_radps = observation.yaw_rate_radps;
  output.turn_mode = turn_mode_;
  output.valid = initialized_;
  return output;
}

void OdometryObserver::update_pose(
  double dt_s, double yaw_rad,
  double previous_body_u_mps, double previous_body_v_mps,
  double body_u_mps, double body_v_mps) noexcept
{
  const double dyaw = wrap_angle(yaw_rad - previous_yaw_rad_);
  const double yaw_mid = wrap_angle(previous_yaw_rad_ + 0.5 * dyaw);
  // The wheel/IMU update below estimates the velocity at the end of this
  // source-time interval. Integrating that endpoint value over the complete
  // interval double-counts launch acceleration and braking. Use the midpoint
  // velocity instead; this is the causal trapezoidal integration of the
  // observer state and is also correct when a stop is detected at this sample.
  const double body_u_mid = 0.5 * (previous_body_u_mps + body_u_mps);
  const double body_v_mid = 0.5 * (previous_body_v_mps + body_v_mps);
  const double vx_world = body_u_mid * std::cos(yaw_mid) -
    body_v_mid * std::sin(yaw_mid);
  const double vy_world = body_u_mid * std::sin(yaw_mid) +
    body_v_mid * std::cos(yaw_mid);
  x_m_ += vx_world * dt_s;
  y_m_ += vy_world * dt_s;
}

void OdometryObserver::update_turn(
  double ax_origin, double ay_origin, double yaw_rate, double dt_s) noexcept
{
  const double du = ax_origin + yaw_rate * body_v_mps_;
  const double dv = ay_origin - yaw_rate * body_u_mps_;
  const double u_mid = body_u_mps_ + 0.5 * dt_s * du;
  const double v_mid = body_v_mps_ + 0.5 * dt_s * dv;
  body_u_mps_ += dt_s * (ax_origin + yaw_rate * v_mid);
  body_v_mps_ += dt_s * (ay_origin - yaw_rate * u_mid);
  body_u_mps_ = std::clamp(body_u_mps_, -30.0, 30.0);
  body_v_mps_ = std::clamp(body_v_mps_, -30.0, 30.0);
  speed_mps_ = std::hypot(body_u_mps_, body_v_mps_);
}

OdometryEstimate OdometryObserver::update(
  const OdometryObservation & observation) noexcept
{
  if (!finite(observation.stamp_s) || !finite(observation.left_angle_rad) ||
    !finite(observation.right_angle_rad) || !finite(observation.ax_mps2) ||
    !finite(observation.ay_mps2) || !finite(observation.yaw_rate_radps) ||
    !finite(observation.yaw_rad))
  {
    auto output = estimate(observation);
    output.timing_degraded = true;
    return output;
  }

  if (!initialized_) {
    initialized_ = true;
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
    body_u_mps_ = 0.0;
    body_v_mps_ = 0.0;
    speed_mps_ = 0.0;
    last_speed_pred_mps_ = 0.0;
    last_wheel_raw_mps_ = 0.0;
    last_wheel_mapped_mps_ = 0.0;
    last_wheel_packet_mps_ = 0.0;
    last_turn_speed_bias_mps_ = 0.0;
    wheel_dropout_active_ = false;
    wheel_burst_rejected_ = false;
    wheel_burst_recovery_pending_ = false;
    encoder_history_.clear();
    encoder_history_.push_back({
      observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
    return estimate(observation);
  }

  const double dt_s = observation.stamp_s - previous_stamp_s_;
  if (dt_s <= 0.0) {
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.timing_degraded = true;
    return output;
  }

  // Preserve the state at the beginning of this source-time interval. The
  // pose update at the end uses it with the newly estimated velocity.
  const double left_delta = observation.left_angle_rad - previous_left_angle_rad_;
  const double right_delta = observation.right_angle_rad - previous_right_angle_rad_;
  const bool encoder_epoch =
    std::abs(left_delta) > config_.reset_encoder_jump_rad ||
    std::abs(right_delta) > config_.reset_encoder_jump_rad;
  if (encoder_epoch) {
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
    turn_mode_ = false;
    turn_calm_time_s_ = 0.0;
    speed_mps_ = 0.0;
    body_u_mps_ = 0.0;
    body_v_mps_ = 0.0;
    previous_pose_body_u_mps_ = 0.0;
    previous_pose_body_v_mps_ = 0.0;
    last_speed_pred_mps_ = 0.0;
    last_wheel_raw_mps_ = 0.0;
    last_wheel_mapped_mps_ = 0.0;
    last_wheel_packet_mps_ = 0.0;
    last_turn_speed_bias_mps_ = 0.0;
    stationary_time_s_ = 0.0;
    wheel_dropout_active_ = false;
    wheel_burst_rejected_ = false;
    wheel_burst_recovery_pending_ = false;
    encoder_history_.clear();
    encoder_history_.push_back({
      observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.reset_epoch = true;
    return output;
  }

  encoder_history_.push_back({
    observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
  EncoderSample wheel_base{
    previous_stamp_s_, previous_left_angle_rad_, previous_right_angle_rad_};
  if (config_.wheel_speed_window_s > 0.0) {
    while (encoder_history_.size() > 1 &&
      observation.stamp_s - encoder_history_[1].stamp_s >= config_.wheel_speed_window_s)
    {
      encoder_history_.pop_front();
    }
    wheel_base = encoder_history_.front();
  }
  const double wheel_window_dt_s = observation.stamp_s - wheel_base.stamp_s;
  const double wheel_raw = wheel_window_dt_s > 0.0 ?
    std::abs(config_.wheel_radius_m * 0.5 *
    ((observation.left_angle_rad - wheel_base.left_angle_rad) +
    (observation.right_angle_rad - wheel_base.right_angle_rad)) /
    wheel_window_dt_s) : 0.0;
  const double wheel_packet = std::abs(config_.wheel_radius_m * 0.5 *
    (left_delta + right_delta) / dt_s);
  // The encoder conversion is the documented wheel radius followed by the
  // identified speed calibration. A table, when configured, corrects the
  // repeatable speed-dependent wheel/body bias without changing the physical
  // simulator or treating wheel spin as body speed.
  const double wheel_mapped = wheel_raw * wheel_scale_for_speed(wheel_raw);
  const double wheel_packet_mapped = wheel_packet * wheel_scale_for_speed(wheel_packet);
  const bool wheel_slew_rejected =
    config_.wheel_speed_slew_limit_mps2 > 0.0 &&
    speed_mps_ > std::max(2.0, config_.wheel_recovery_launch_speed_mps) &&
    last_wheel_mapped_mps_ > 0.0 &&
    std::abs(wheel_mapped - last_wheel_mapped_mps_) / dt_s >
    config_.wheel_speed_slew_limit_mps2 &&
    std::abs(wheel_mapped - speed_mps_) > config_.wheel_innovation_max_mps;
  last_wheel_raw_mps_ = wheel_raw;
  last_wheel_mapped_mps_ = wheel_mapped;
  last_wheel_packet_mps_ = wheel_packet;
  last_turn_speed_bias_mps_ = 0.0;

  // A normal acceleration packet raises both estimates together. The
  // simulator's delayed cumulative encoder packet instead raises the rolling
  // estimate above the previous causal speed and then reports an even larger
  // instantaneous packet. Do not let either part of that burst update speed;
  // the existing dropout path propagates the last causal speed with IMU data
  // until a coherent packet returns.
  wheel_burst_rejected_ = wheel_slew_rejected ||
    (config_.wheel_burst_disagreement_mps > 0.0 &&
    wheel_mapped > speed_mps_ + config_.wheel_burst_disagreement_mps &&
    wheel_packet_mapped > wheel_mapped + config_.wheel_burst_disagreement_mps);

  // A repeated cumulative-angle packet can land just above the absolute
  // frozen-wheel threshold (the recorded failure was 0.151 m/s against a
  // causal speed above 4 m/s).  In that case both the rolling and packet
  // rates agree with each other, so the coherence shortcut below would
  // incorrectly accept the missing-motion sample and collapse odometry.
  // Treat a near-zero packet rate as a dropout regardless of the absolute
  // freeze threshold. The 0.25*speed term scales the test at low speed while
  // the 0.5 m/s floor catches the recorded 0.151 m/s packet.
  const double near_zero_packet_limit_mps = std::max(
    0.5, 0.25 * speed_mps_);
  const bool moving_encoder_dropout =
    speed_mps_ > 0.5 && wheel_packet_mapped < near_zero_packet_limit_mps;
  if (moving_encoder_dropout) {
    wheel_dropout_active_ = true;
  }

  // A repeated cumulative encoder angle is a missing-motion packet, even
  // though the longer window still contains motion from earlier packets.
  // Remember it until a current packet agrees with the IMU prediction; this
  // prevents the window estimate from erasing distance during recovery.
  if (wheel_packet < config_.wheel_freeze_speed_mps && speed_mps_ > 0.5) {
    wheel_dropout_active_ = true;
  }
  if (wheel_burst_rejected_) {
    wheel_dropout_active_ = true;
    wheel_burst_recovery_pending_ = true;
  }

  // A repeated cumulative-encoder burst can make the rolling and current
  // packet rates agree while both are far above the causal body speed. The
  // coherence shortcut below is useful for a stale low window, but it must
  // not promote this physically impossible positive jump when a steering
  // transient enters turn mode. Hold the causal state and let IMU
  // propagation catch up until a wheel packet returns inside the innovation
  // gate.
  const bool positive_wheel_innovation_fault =
    config_.wheel_innovation_max_mps > 0.0 &&
    speed_mps_ >= std::max(2.0, config_.wheel_recovery_launch_speed_mps) &&
    wheel_mapped > speed_mps_ + config_.wheel_innovation_max_mps &&
    wheel_packet_mapped > speed_mps_ + config_.wheel_innovation_max_mps &&
    config_.wheel_burst_disagreement_mps > 0.0 &&
    std::abs(wheel_packet_mapped - wheel_mapped) <=
    0.25 * config_.wheel_burst_disagreement_mps &&
    !wheel_burst_rejected_;
  if (positive_wheel_innovation_fault) {
    wheel_dropout_active_ = true;
    wheel_burst_recovery_pending_ = true;
  }

  // A coherent rolling-window/current-packet pair is not sufficient evidence
  // during launch: both rates can describe wheel spin while the body is still
  // accelerating from rest. The accepted model-identification recordings
  // repeatedly show approximately 6 m/s wheel rates below 1 m/s body speed.
  // Keep this gate limited to the low-speed recovery shortcut; established
  // motion and ordinary innovation checks are unchanged.
  const auto launch_wheel_spin = [&](double predicted_speed) {
    return config_.wheel_recovery_launch_speed_mps > 0.0 &&
           config_.wheel_recovery_launch_innovation_mps > 0.0 &&
           config_.wheel_recovery_launch_wheel_speed_mps > 0.0 &&
           predicted_speed < config_.wheel_recovery_launch_speed_mps &&
           wheel_packet_mapped > config_.wheel_recovery_launch_wheel_speed_mps &&
           wheel_packet_mapped > predicted_speed +
           config_.wheel_recovery_launch_innovation_mps;
  };

  // A degraded packet is not necessarily a missing-motion packet. The
  // synchronized encoder endpoints still describe the average displacement
  // across a short gap. The old early return deleted that displacement and
  // created a repeatable odometry error on the track. Rebaseline only for a
  // genuinely long gap; short gaps continue through the normal wheel gate.
  if (dt_s > config_.degraded_packet_dt_max_s &&
    dt_s > config_.max_integratable_gap_s)
  {
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
    encoder_history_.clear();
    encoder_history_.push_back({
      observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
    last_speed_pred_mps_ = speed_mps_;
    previous_pose_body_u_mps_ = body_u_mps_;
    previous_pose_body_v_mps_ = body_v_mps_;
    stationary_time_s_ = 0.0;
    wheel_dropout_active_ = false;
    wheel_burst_rejected_ = false;
    wheel_burst_recovery_pending_ = false;
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.timing_degraded = true;
    return output;
  }

  const auto wheel_speed_is_valid = [&](double predicted_speed) {
    if ((dt_s > config_.normal_packet_dt_max_s &&
      dt_s > config_.max_integratable_gap_s) || !finite(wheel_mapped)) {
      return false;
    }
    if (launch_wheel_spin(predicted_speed)) {
      return false;
    }
    if (wheel_slew_rejected) {
      return false;
    }
    // A zero encoder packet is allowed to brake a stopped/slow estimate, but
    // not to erase a moving estimate after one missing callback.
    if (wheel_raw < config_.wheel_freeze_speed_mps && predicted_speed > 0.5) {
      return false;
    }
    // Permit the first positive wheel sample to establish motion.  The old
    // innovation gate compared it with a zero IMU prediction and rejected
    // the launch sample, which was especially harmful when turn mode was
    // entered immediately by the steering transient.
    return std::abs(wheel_mapped - predicted_speed) <=
      config_.wheel_innovation_max_mps ||
      (predicted_speed < config_.wheel_freeze_speed_mps &&
      wheel_mapped >= config_.stationary_speed_threshold_mps);
  };

  // The simulator has emitted an impossible longitudinal acceleration sample
  // while stationary (approximately -123 m/s^2).  In turn mode that value
  // would be integrated as a real backwards velocity before the wheel gate
  // can reject it.  Re-baseline the packet, hold the last causal state, and
  // mark the sample degraded instead.  The raw IMU value remains available
  // through the recorder for diagnosis.
  if (std::abs(observation.ax_mps2) > config_.max_imu_ax_abs_mps2) {
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
    last_speed_pred_mps_ = speed_mps_;
    stationary_time_s_ = 0.0;
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.timing_degraded = true;
    output.sensor_outlier = true;
    return output;
  }

  const bool calm_stationary_sample =
    wheel_raw < config_.stationary_speed_threshold_mps &&
    std::abs(observation.ax_mps2) <= config_.stationary_ax_abs_max_mps2 &&
    std::abs(observation.ay_mps2) <= config_.stationary_ay_abs_max_mps2 &&
    std::abs(observation.yaw_rate_radps) <=
    config_.stationary_yaw_rate_abs_max_radps;
  stationary_time_s_ = calm_stationary_sample ?
    stationary_time_s_ + dt_s : 0.0;

  // The normal frozen-wheel gate protects against a single missing encoder
  // packet.  It must not turn a stopped/collided car into a permanently
  // moving car, though: once the wheel and IMU have been calm for the hold
  // interval, force a zero-speed epoch and publish it immediately.
  if (stationary_time_s_ >= config_.stationary_hold_s) {
    speed_mps_ = 0.0;
    body_u_mps_ = 0.0;
    body_v_mps_ = 0.0;
    turn_mode_ = false;
    turn_calm_time_s_ = 0.0;
    last_speed_pred_mps_ = 0.0;
    update_pose(
      dt_s, observation.yaw_rad,
      previous_pose_body_u_mps_, previous_pose_body_v_mps_,
      body_u_mps_, body_v_mps_);
    previous_pose_body_u_mps_ = body_u_mps_;
    previous_pose_body_v_mps_ = body_v_mps_;
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
    wheel_dropout_active_ = false;
    auto output = estimate(observation);
    output.dt_s = dt_s;
    return output;
  }

  const double yaw_alpha = (observation.yaw_rate_radps - previous_yaw_rate_radps_) / dt_s;
  const double ax_origin = observation.ax_mps2 +
    observation.yaw_rate_radps * observation.yaw_rate_radps * config_.imu_x_offset_m;
  const double ay_origin = observation.ay_mps2 - yaw_alpha * config_.imu_x_offset_m;

  if (!turn_mode_ &&
    (std::abs(observation.yaw_rate_radps) >= config_.turn_enter_yaw_rate_radps ||
    std::abs(observation.ay_mps2) >= config_.turn_enter_abs_ay_mps2))
  {
    const double turn_wheel_bias = turn_speed_bias_mps(
      wheel_mapped, observation.yaw_rate_radps);
    turn_mode_ = true;
    turn_calm_time_s_ = 0.0;
    // The first turn can be entered during launch, before the IMU speed
    // prediction has caught up with the synchronized encoder packet. The
    // encoder is the trusted longitudinal measurement in turn mode; keeping
    // the lower prediction here makes the normal innovation gate reject valid
    // wheel speeds for several packets and loses launch distance.
    if (!wheel_dropout_active_ &&
      wheel_packet >= config_.wheel_freeze_speed_mps && finite(wheel_mapped)) {
      body_u_mps_ = std::max(0.0, wheel_mapped + turn_wheel_bias);
      speed_mps_ = body_u_mps_;
    } else {
      body_u_mps_ = speed_mps_;
    }
    body_v_mps_ = 0.0;
  }

  bool wheel_update_used = false;
  double pose_body_u_mps = body_u_mps_;
  double pose_body_v_mps = body_v_mps_;
  double speed_pred = speed_mps_;
  if (turn_mode_) {
    const double turn_wheel_bias = turn_speed_bias_mps(
      wheel_mapped, observation.yaw_rate_radps);
    const double turn_wheel_mapped = std::max(0.0, wheel_mapped + turn_wheel_bias);
    const double turn_wheel_packet_mapped = std::max(
      0.0, wheel_packet_mapped + turn_speed_bias_mps(
      wheel_packet_mapped, observation.yaw_rate_radps));
    last_turn_speed_bias_mps_ = turn_wheel_bias;
    const bool integrate_lateral_dynamics =
      config_.integrate_lateral_acceleration_in_turn || wheel_dropout_active_;
    // Longitudinal wheel speed is the observable with the correct scale in a
    // turn.  Use it to anchor u before and after the IMU RK2 lateral update;
    // otherwise the turn model integrates small ax bias and can report
    // 0.30 m/s while the encoders and vehicle are travelling at 0.50 m/s.
    const bool normal_wheel_recovery = wheel_dropout_active_ &&
      !wheel_burst_rejected_ &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      (!wheel_burst_recovery_pending_ ||
      (config_.wheel_burst_disagreement_mps > 0.0 &&
      std::abs(turn_wheel_packet_mapped - turn_wheel_mapped) <=
      config_.wheel_burst_disagreement_mps)) &&
      (std::abs(turn_wheel_packet_mapped - speed_mps_) <=
      config_.wheel_innovation_max_mps) &&
      turn_wheel_packet_mapped <= speed_mps_ +
      config_.turn_current_packet_max_increase_mps;
    const bool turn_current_packet_recovery =
      config_.allow_turn_current_packet_recovery && wheel_dropout_active_ &&
      wheel_burst_recovery_pending_ && !wheel_burst_rejected_ &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      turn_wheel_packet_mapped >= speed_mps_ &&
      turn_wheel_packet_mapped <= speed_mps_ +
      config_.turn_current_packet_max_increase_mps &&
      config_.wheel_burst_disagreement_mps > 0.0 &&
      turn_wheel_packet_mapped > turn_wheel_mapped +
      config_.wheel_burst_disagreement_mps;
    const bool wheel_recovery = normal_wheel_recovery ||
      turn_current_packet_recovery;
    const bool wheel_coherent = !wheel_burst_rejected_ &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      config_.wheel_burst_disagreement_mps > 0.0 &&
      !wheel_slew_rejected &&
      (speed_mps_ < std::max(2.0, config_.wheel_recovery_launch_speed_mps) ||
      std::abs(turn_wheel_mapped - speed_mps_) <=
      config_.wheel_innovation_max_mps) &&
      std::abs(turn_wheel_packet_mapped - turn_wheel_mapped) <=
      config_.wheel_burst_disagreement_mps;
    const bool wheel_ok = !wheel_burst_rejected_ && !wheel_dropout_active_ &&
      (wheel_speed_is_valid(speed_mps_) || wheel_coherent);
    if (wheel_ok) {
      body_u_mps_ = turn_wheel_mapped;
      wheel_update_used = true;
    } else if (wheel_recovery) {
      body_u_mps_ = turn_wheel_packet_mapped;
      wheel_dropout_active_ = false;
      wheel_burst_recovery_pending_ = false;
      // The rolling window still contains the repeated cumulative-angle
      // packet. Rebase it at the first valid recovery packet so the next
      // normal update cannot lock onto a stale low rate.
      encoder_history_.clear();
      encoder_history_.push_back({
        observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
      wheel_update_used = true;
    } else if (wheel_dropout_active_ && !integrate_lateral_dynamics) {
      // The encoder stream can repeat one cumulative angle for several source
      // packets while the car is braking in a turn. Propagate the last causal
      // longitudinal speed with the IMU until a coherent encoder displacement
      // returns; using the lagging window here loses distance.
      double braking_ax = observation.ax_mps2;
      if (observation.ax_mps2 < config_.decel_detect_ax_mps2) {
        braking_ax = config_.decel_ax_scale * observation.ax_mps2 +
          config_.decel_ax_offset_mps2;
      }
      braking_ax += observation.yaw_rate_radps * observation.yaw_rate_radps *
        config_.imu_x_offset_m;
      // Positive ax is ambiguous while wheel slip is active because the
      // unobserved r*v term can have either sign. Apply only deceleration and
      // hold through positive acceleration until a causal wheel sample returns.
      if (braking_ax < 0.0) {
        body_u_mps_ = std::max(0.0, body_u_mps_ + braking_ax * dt_s);
      }
    } else if (!integrate_lateral_dynamics &&
      wheel_raw < config_.wheel_freeze_speed_mps &&
      observation.ax_mps2 <= config_.turn_wheel_braking_ax_mps2)
    {
      // The encoder stream can repeat one cumulative angle for several
      // source packets while the car is braking in a turn.  Holding the old
      // speed in that case creates a large forward-pose error.  A strongly
      // negative longitudinal IMU sample is the useful measurement for this
      // specific missing-encoder case; isolated zero packets without braking
      // remain protected by the frozen-wheel gate above.  Do not apply this
      // correction when the dropout path already selected the RK2 dynamic
      // update: that update consumes the same longitudinal acceleration.
      double braking_ax = observation.ax_mps2;
      if (observation.ax_mps2 < config_.decel_detect_ax_mps2) {
        braking_ax = config_.decel_ax_scale * observation.ax_mps2 +
          config_.decel_ax_offset_mps2;
      }
      braking_ax += observation.yaw_rate_radps * observation.yaw_rate_radps *
        config_.imu_x_offset_m;
      if (braking_ax < 0.0) {
        body_u_mps_ = std::max(0.0, body_u_mps_ + braking_ax * dt_s);
      }
    }
    if (integrate_lateral_dynamics) {
      // A repeated encoder packet is a longitudinal measurement dropout,
      // not permission to bypass the calibrated braking path.  The RK2 turn
      // integrator is still needed for the lateral state, but using raw ax
      // here made the configured deceleration calibration effective only in
      // straight mode and in the unreachable non-dynamic dropout branch.
      double turn_ax_origin = ax_origin;
      if (wheel_dropout_active_ && observation.ax_mps2 < config_.decel_detect_ax_mps2) {
        turn_ax_origin = config_.decel_ax_scale * observation.ax_mps2 +
          config_.decel_ax_offset_mps2 +
          observation.yaw_rate_radps * observation.yaw_rate_radps *
          config_.imu_x_offset_m;
      }
      update_turn(turn_ax_origin, ay_origin, observation.yaw_rate_radps, dt_s);
      if (wheel_ok || wheel_recovery) {
        // A recovery packet is a fresh synchronized wheel measurement. The
        // dynamic update above is still needed for the lateral state, but it
        // must not overwrite the recovered longitudinal anchor with the same
        // interval's IMU acceleration. Otherwise a delayed-encoder burst
        // creates an artificial under-speed step immediately after recovery.
        body_u_mps_ = wheel_recovery ? turn_wheel_packet_mapped : turn_wheel_mapped;
      }
    } else {
      // The encoder is the trusted longitudinal measurement in a turn. Do
      // not integrate the lateral IMU sample into a persistent pose velocity:
      // source-time replay showed that this creates false lateral displacement
      // in the first corner. A separately identified, bounded sideslip model
      // is allowed here because it uses only causal wheel speed and yaw rate.
      body_v_mps_ = kinematic_lateral_velocity(
        observation.yaw_rate_radps, body_u_mps_);
    }
    const bool coherent_current_packet =
      wheel_update_used && !wheel_burst_rejected_ &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      config_.wheel_burst_disagreement_mps > 0.0 &&
      std::abs(turn_wheel_packet_mapped - turn_wheel_mapped) <=
      config_.wheel_burst_disagreement_mps;
    if (config_.use_coherent_packet_velocity_for_pose &&
      coherent_current_packet && finite(wheel_packet_mapped))
    {
      const double blend = std::clamp(
        config_.coherent_packet_pose_blend, 0.0, 1.0);
      pose_body_u_mps = (1.0 - blend) * body_u_mps_ +
        blend * turn_wheel_packet_mapped;
      pose_body_v_mps = body_v_mps_;
    } else {
      pose_body_u_mps = body_u_mps_;
      pose_body_v_mps = body_v_mps_;
    }
    speed_mps_ = std::hypot(body_u_mps_, body_v_mps_);
    speed_pred = speed_mps_;
    const bool calm =
      std::abs(observation.yaw_rate_radps) < config_.turn_exit_yaw_rate_radps &&
      std::abs(observation.ay_mps2) < config_.turn_exit_abs_ay_mps2;
    turn_calm_time_s_ = calm ? turn_calm_time_s_ + dt_s : 0.0;
    if (turn_calm_time_s_ >= config_.turn_exit_hold_s) {
      speed_mps_ = std::hypot(body_u_mps_, body_v_mps_);
      body_u_mps_ = speed_mps_;
      body_v_mps_ = 0.0;
      turn_mode_ = false;
      turn_calm_time_s_ = 0.0;
      pose_body_u_mps = body_u_mps_;
      pose_body_v_mps = body_v_mps_;
    }
  } else {
    double ax_effective = observation.ax_mps2;
    if (observation.ax_mps2 < config_.decel_detect_ax_mps2) {
      ax_effective = config_.decel_ax_scale * observation.ax_mps2 +
        config_.decel_ax_offset_mps2;
    }
    speed_pred = std::max(0.0, speed_mps_ + ax_effective * dt_s);
    speed_mps_ = speed_pred;
    // A source interval above the nominal 35 ms contract is degraded for
    // diagnostics, but its synchronized encoder endpoints are still useful
    // while the gap remains inside the integratable horizon.
    if (dt_s <= config_.max_integratable_gap_s) {
      const bool wheel_recovery = wheel_dropout_active_ &&
        !wheel_burst_rejected_ &&
        !launch_wheel_spin(speed_pred) &&
        wheel_packet >= config_.wheel_freeze_speed_mps &&
        (!wheel_burst_recovery_pending_ ||
        (config_.wheel_burst_disagreement_mps > 0.0 &&
        std::abs(wheel_packet_mapped - wheel_mapped) <=
        config_.wheel_burst_disagreement_mps)) &&
        (std::abs(wheel_packet_mapped - speed_pred) <=
        config_.wheel_innovation_max_mps) &&
        (!wheel_burst_recovery_pending_ ||
        config_.turn_current_packet_max_increase_mps <= 0.0 ||
        wheel_packet_mapped <= speed_pred +
        config_.turn_current_packet_max_increase_mps);
      const bool wheel_ok =
        std::abs(observation.ax_mps2) < config_.wheel_update_ax_abs_max_mps2 &&
        (!wheel_dropout_active_ ?
        (wheel_speed_is_valid(speed_pred) ||
        (!wheel_burst_rejected_ &&
        !launch_wheel_spin(speed_pred) &&
        !wheel_slew_rejected &&
        wheel_packet >= config_.wheel_freeze_speed_mps &&
        config_.wheel_burst_disagreement_mps > 0.0 &&
        std::abs(wheel_packet_mapped - wheel_mapped) <=
        config_.wheel_burst_disagreement_mps)) : wheel_recovery);
      if (wheel_ok) {
        if (wheel_recovery) {
          speed_mps_ = wheel_packet_mapped;
          wheel_dropout_active_ = false;
          wheel_burst_recovery_pending_ = false;
          // Discard the repeated-angle sample from the rolling window. The
          // recovered current packet is the new causal encoder baseline.
          encoder_history_.clear();
          encoder_history_.push_back({
            observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
        } else {
          speed_mps_ = speed_pred < config_.wheel_freeze_speed_mps ? wheel_mapped :
            (1.0 - config_.wheel_update_beta) * speed_pred +
            config_.wheel_update_beta * wheel_mapped;
        }
        wheel_update_used = true;
      }
    }
    body_u_mps_ = speed_mps_;
    body_v_mps_ = 0.0;
    pose_body_u_mps = body_u_mps_;
    pose_body_v_mps = body_v_mps_;
  }

  last_speed_pred_mps_ = speed_pred;
  update_pose(
    dt_s, observation.yaw_rad,
    previous_pose_body_u_mps_, previous_pose_body_v_mps_,
    pose_body_u_mps, pose_body_v_mps);
  previous_pose_body_u_mps_ = pose_body_u_mps;
  previous_pose_body_v_mps_ = pose_body_v_mps;
  previous_stamp_s_ = observation.stamp_s;
  previous_left_angle_rad_ = observation.left_angle_rad;
  previous_right_angle_rad_ = observation.right_angle_rad;
  previous_yaw_rad_ = observation.yaw_rad;
  previous_yaw_rate_radps_ = observation.yaw_rate_radps;

  auto output = estimate(observation);
  output.dt_s = dt_s;
  output.wheel_update_used = wheel_update_used;
  output.timing_degraded = dt_s > config_.normal_packet_dt_max_s;
  return output;
}

}  // namespace f1tenth_localization

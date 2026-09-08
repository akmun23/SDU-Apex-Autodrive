#include "f1tenth_localization/odometry_observer.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "f1tenth_localization/wheel_speed_map.hpp"

namespace f1tenth_localization
{

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
  x_m_ = 0.0;
  y_m_ = 0.0;
  last_speed_pred_mps_ = 0.0;
  last_wheel_raw_mps_ = 0.0;
  last_wheel_mapped_mps_ = 0.0;
  stationary_time_s_ = 0.0;
}

double OdometryObserver::wrap_angle(double angle) noexcept
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

bool OdometryObserver::finite(double value) noexcept
{
  return std::isfinite(value);
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
  output.ax_mps2 = observation.ax_mps2;
  output.ay_mps2 = observation.ay_mps2;
  output.yaw_rate_radps = observation.yaw_rate_radps;
  output.turn_mode = turn_mode_;
  output.valid = initialized_;
  return output;
}

void OdometryObserver::update_pose(double dt_s, double yaw_rad) noexcept
{
  const double dyaw = wrap_angle(yaw_rad - previous_yaw_rad_);
  const double yaw_mid = wrap_angle(previous_yaw_rad_ + 0.5 * dyaw);
  const double vx_world = body_u_mps_ * std::cos(yaw_mid) -
    body_v_mps_ * std::sin(yaw_mid);
  const double vy_world = body_u_mps_ * std::sin(yaw_mid) +
    body_v_mps_ * std::cos(yaw_mid);
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
    return estimate(observation);
  }

  const double dt_s = observation.stamp_s - previous_stamp_s_;
  if (dt_s <= 0.0) {
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.timing_degraded = true;
    return output;
  }

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
    last_speed_pred_mps_ = 0.0;
    last_wheel_raw_mps_ = 0.0;
    last_wheel_mapped_mps_ = 0.0;
    stationary_time_s_ = 0.0;
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.reset_epoch = true;
    return output;
  }

  if (dt_s > config_.degraded_packet_dt_max_s) {
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
    last_speed_pred_mps_ = speed_mps_;
    last_wheel_raw_mps_ = 0.0;
    last_wheel_mapped_mps_ = 0.0;
    stationary_time_s_ = 0.0;
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.timing_degraded = true;
    return output;
  }

  const double wheel_raw = std::abs(config_.wheel_radius_m * 0.5 *
    (left_delta + right_delta) / dt_s);
  const double wheel_mapped = WheelSpeedMap::map(wheel_raw);
  last_wheel_raw_mps_ = wheel_raw;
  last_wheel_mapped_mps_ = wheel_mapped;

  const auto wheel_speed_is_valid = [&](double predicted_speed) {
    if (dt_s > config_.normal_packet_dt_max_s || !finite(wheel_mapped)) {
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
      wheel_mapped >= config_.wheel_freeze_speed_mps);
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
    update_pose(dt_s, observation.yaw_rad);
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
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
    turn_mode_ = true;
    turn_calm_time_s_ = 0.0;
    body_u_mps_ = speed_mps_;
    body_v_mps_ = 0.0;
  }

  bool wheel_update_used = false;
  double speed_pred = speed_mps_;
  if (turn_mode_) {
    // Longitudinal wheel speed is the observable with the correct scale in a
    // turn.  Use it to anchor u before and after the IMU RK2 lateral update;
    // otherwise the turn model integrates small ax bias and can report
    // 0.30 m/s while the encoders and vehicle are travelling at 0.50 m/s.
    const bool wheel_ok = wheel_speed_is_valid(speed_mps_);
    if (wheel_ok) {
      body_u_mps_ = wheel_mapped;
      wheel_update_used = true;
    }
    if (config_.integrate_lateral_acceleration_in_turn) {
      update_turn(ax_origin, ay_origin, observation.yaw_rate_radps, dt_s);
      if (wheel_ok) {
        body_u_mps_ = wheel_mapped;
      }
    } else {
      // The encoder is the trusted longitudinal measurement in a turn. Do
      // not integrate the simulator's lateral IMU sample into a persistent
      // pose velocity: source-time replay showed that this creates several
      // centimetres of false lateral displacement within the first corner.
      body_v_mps_ = 0.0;
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
    }
  } else {
    double ax_effective = observation.ax_mps2;
    if (observation.ax_mps2 < config_.decel_detect_ax_mps2) {
      ax_effective = config_.decel_ax_scale * observation.ax_mps2 +
        config_.decel_ax_offset_mps2;
    }
    speed_pred = std::max(0.0, speed_mps_ + ax_effective * dt_s);
    speed_mps_ = speed_pred;
    if (dt_s <= config_.normal_packet_dt_max_s) {
      const bool wheel_ok =
        std::abs(observation.ax_mps2) < config_.wheel_update_ax_abs_max_mps2 &&
        wheel_speed_is_valid(speed_pred);
      if (wheel_ok) {
        speed_mps_ = speed_pred < config_.wheel_freeze_speed_mps ?
          wheel_mapped :
          (1.0 - config_.wheel_update_beta) * speed_pred +
          config_.wheel_update_beta * wheel_mapped;
        wheel_update_used = true;
      }
    }
    body_u_mps_ = speed_mps_;
    body_v_mps_ = 0.0;
  }

  last_speed_pred_mps_ = speed_pred;
  update_pose(dt_s, observation.yaw_rad);
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

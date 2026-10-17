#include "f1tenth_localization/odometry_observer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>


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
  config.decel_detect_ax_mps2 = -0.5;
  config.decel_ax_scale = 1.005;
  config.decel_ax_offset_mps2 = 0.020;
  config.wheel_dropout_positive_ax_max_mps2 = 4.4;
  config.wheel_update_ax_abs_max_mps2 = 6.5;
  config.wheel_freeze_speed_mps = 0.15;
  config.wheel_recovery_launch_speed_mps = 2.0;
  config.wheel_recovery_launch_innovation_mps = 2.0;
  config.wheel_recovery_launch_wheel_speed_mps = 4.0;
  config.wheel_burst_disagreement_mps = 1.0;
  config.wheel_burst_catchup_accel_mps2 = 6.5;
  config.wheel_burst_catchup_max_mps = 16.0;
  config.allow_turn_current_packet_recovery = true;
  config.turn_current_packet_max_increase_mps = 0.20;
  config.use_turn_speed_bias_model = true;
  config.turn_speed_bias_constant_mps = -0.03;
  config.turn_speed_bias_speed_mps = 0.0;
  config.turn_speed_bias_speed_squared_mps = 0.0;
  config.turn_speed_bias_yaw_rate_abs_mps = 0.185;
  config.turn_speed_bias_yaw_rate_squared_mps = 0.0;
  config.turn_speed_bias_speed_yaw_rate_abs_mps = 0.0;
  config.turn_speed_bias_max_mps = 0.15;
  config.turn_speed_bias_start_speed_mps = 6.5;
  config.turn_speed_bias_full_speed_mps = 7.5;
  config.use_turn_speed_bias_lateral_accel_gate = true;
  config.turn_speed_bias_lateral_accel_gate_start_mps2 = 3.0;
  config.turn_speed_bias_lateral_accel_gate_full_mps2 = 5.0;
  config.turn_slip_speed_bias_lateral_accel_threshold_mps2 = 4.0;
  config.turn_slip_speed_bias_gain_mps_per_mps2 = 0.05;
  config.turn_slip_speed_bias_start_speed_mps = 2.0;
  config.turn_slip_speed_bias_full_speed_mps = 2.8;
  config.turn_slip_speed_bias_fade_start_mps = 4.0;
  config.turn_slip_speed_bias_fade_end_mps = 4.5;
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
  config.turn_wheel_update_start_speed_mps = 7.0;
  config.turn_wheel_update_min_ax_mps2 = 0.0;
  config.turn_wheel_update_innovation_threshold_mps = 0.0;
  config.turn_wheel_update_slip_beta = 0.125;
  config.use_high_yaw_wheel_slip_observer = false;
  config.high_yaw_wheel_slip_enter_radps = 2.5;
  config.high_yaw_wheel_slip_exit_radps = 1.5;
  config.high_yaw_wheel_slip_exit_hold_s = 0.10;
  config.high_yaw_wheel_slip_innovation_mps = 2.0;
  config.use_learned_u_residual_model = false;
  config.learned_u_residual_min_speed_mps = 2.0;
  config.learned_u_residual_max_speed_mps = 8.0;
  config.u_correction_support_speed_edges_mps = {2.0, 4.0, 6.0, 8.0};
  config.u_correction_support_max_abs_steering_rad = {0.30, 0.20, 0.10};
  config.learned_u_residual_intercept_mps = 0.060340928230651916;
  config.learned_u_residual_coefficients = {
    -0.01123104527731476, -0.9252468532458303,
    -0.0023773279467559755, -0.0008304085906844171,
    -0.00922115794747585, 0.013095896837034868,
    -0.127764062201128, 0.7369086293590875,
    0.43005743300167854, 0.054493639178485584,
    -0.06106141502377045, -0.022945993984350592,
    -0.010203043671907322, -0.009358420157824551,
    0.00494236405039215, 0.007354863850582028,
    0.4276528797732827, -0.14567326853467916,
    0.05938704203530415, 0.018871648829481173,
    0.043191576713956605, -0.7051323076439538};
  config.use_unity_actuator_u_experts = false;
  config.unity_actuator_u_min_speed_mps = 2.0;
  config.unity_actuator_u_max_speed_mps = 8.0;
  config.unity_zero_throttle_intercept_mps = 0.0;
  config.unity_nonzero_throttle_intercept_mps = 0.0;
  config.use_kinematic_lateral_slip_model = true;
  config.lateral_velocity_yaw_rate_gain_m = 0.181939;
  config.lateral_velocity_speed_yaw_rate_gain_s = -0.009656;
  config.lateral_velocity_reference_forward_offset_m = 0.15532;
  config.lateral_velocity_max_mps = 0.35;
  config.max_imu_ax_abs_mps2 = 30.0;
  config.imu_acceleration_reference_x_m = 0.15532;
  config.physx_velocity_blend = 0.0;
  return config;
}

OdometryObserver::OdometryObserver(OdometryObserverConfig config)
: config_(config)
{
  if (!std::isfinite(config_.physx_velocity_blend) ||
    config_.physx_velocity_blend < 0.0 ||
    config_.physx_velocity_blend > 1.0)
  {
    throw std::invalid_argument("PhysX Odom velocity blend must be in [0, 1]");
  }
  if (!std::isfinite(config_.physx_forward_velocity_blend) ||
    config_.physx_forward_velocity_blend < 0.0 ||
    config_.physx_forward_velocity_blend > 1.0)
  {
    throw std::invalid_argument("PhysX forward velocity blend must be in [0, 1]");
  }
  if (!std::isfinite(config_.physx_low_speed_velocity_blend) ||
    config_.physx_low_speed_velocity_blend < 0.0 ||
    config_.physx_low_speed_velocity_blend > 1.0 ||
    !std::isfinite(config_.physx_low_speed_blend_transition_mps) ||
    config_.physx_low_speed_blend_transition_mps <= 0.0)
  {
    throw std::invalid_argument("invalid low-speed PhysX Odom blend schedule");
  }
  if (config_.use_learned_u_residual_model &&
    (config_.learned_u_residual_coefficients.size() != 22 ||
    !std::isfinite(config_.learned_u_residual_min_speed_mps) ||
    !std::isfinite(config_.learned_u_residual_max_speed_mps) ||
    config_.learned_u_residual_min_speed_mps >=
    config_.learned_u_residual_max_speed_mps ||
    !std::isfinite(config_.learned_u_residual_intercept_mps) ||
    !std::all_of(
      config_.learned_u_residual_coefficients.begin(),
      config_.learned_u_residual_coefficients.end(),
      [](double value) {return std::isfinite(value);})))
  {
    throw std::invalid_argument("invalid enabled Odom-u residual model parameters");
  }
  if (config_.use_unity_actuator_u_experts &&
    (config_.unity_zero_throttle_coefficients.size() != 34 ||
    config_.unity_nonzero_throttle_coefficients.size() != 34 ||
    !std::isfinite(config_.unity_actuator_u_min_speed_mps) ||
    !std::isfinite(config_.unity_actuator_u_max_speed_mps) ||
    config_.unity_actuator_u_min_speed_mps >=
    config_.unity_actuator_u_max_speed_mps ||
    !std::isfinite(config_.unity_zero_throttle_intercept_mps) ||
    !std::isfinite(config_.unity_nonzero_throttle_intercept_mps) ||
    !std::all_of(
      config_.unity_zero_throttle_coefficients.begin(),
      config_.unity_zero_throttle_coefficients.end(),
      [](double value) {return std::isfinite(value);}) ||
    !std::all_of(
      config_.unity_nonzero_throttle_coefficients.begin(),
      config_.unity_nonzero_throttle_coefficients.end(),
      [](double value) {return std::isfinite(value); })))
  {
    throw std::invalid_argument("invalid enabled Unity actuator Odom model parameters");
  }
  if (config_.use_high_steer_unity_actuator_u_expert &&
    (config_.high_steer_unity_actuator_u_coefficients.size() != 34 ||
    !std::isfinite(config_.high_steer_unity_actuator_u_min_speed_mps) ||
    !std::isfinite(config_.high_steer_unity_actuator_u_max_speed_mps) ||
    !std::isfinite(config_.high_steer_unity_actuator_u_max_speed_low_throttle_mps) ||
    !std::isfinite(config_.high_steer_unity_actuator_u_throttle_split) ||
    config_.high_steer_unity_actuator_u_min_speed_mps >=
    config_.high_steer_unity_actuator_u_max_speed_mps ||
    config_.high_steer_unity_actuator_u_max_speed_low_throttle_mps <=
    config_.high_steer_unity_actuator_u_min_speed_mps ||
    config_.high_steer_unity_actuator_u_max_speed_low_throttle_mps >
    config_.high_steer_unity_actuator_u_max_speed_mps ||
    !std::isfinite(config_.high_steer_unity_actuator_u_min_abs_yaw_rate_radps) ||
    config_.high_steer_unity_actuator_u_min_abs_yaw_rate_radps < 0.0 ||
    !std::isfinite(config_.high_steer_unity_actuator_u_min_abs_steering_rad) ||
    !std::isfinite(config_.high_steer_unity_actuator_u_max_abs_steering_rad) ||
    config_.high_steer_unity_actuator_u_min_abs_steering_rad >=
    config_.high_steer_unity_actuator_u_max_abs_steering_rad ||
    !std::isfinite(config_.high_steer_unity_actuator_u_min_throttle_feedback) ||
    !std::isfinite(config_.high_steer_unity_actuator_u_max_throttle_feedback) ||
    config_.high_steer_unity_actuator_u_min_throttle_feedback >
    config_.high_steer_unity_actuator_u_max_throttle_feedback ||
    config_.high_steer_unity_actuator_u_throttle_split <
    config_.high_steer_unity_actuator_u_min_throttle_feedback ||
    config_.high_steer_unity_actuator_u_throttle_split >
    config_.high_steer_unity_actuator_u_max_throttle_feedback ||
    !std::isfinite(config_.high_steer_unity_actuator_u_intercept_mps) ||
    !std::all_of(
      config_.high_steer_unity_actuator_u_coefficients.begin(),
      config_.high_steer_unity_actuator_u_coefficients.end(),
      [](double value) {return std::isfinite(value);})))
  {
    throw std::invalid_argument("invalid enabled high-steer Odom-u expert parameters");
  }
  const auto & support_edges = config_.u_correction_support_speed_edges_mps;
  const auto & support_steering = config_.u_correction_support_max_abs_steering_rad;
  if (support_edges.size() != support_steering.size() + 1 ||
    support_edges.size() < 2 ||
    !std::all_of(support_edges.begin(), support_edges.end(),
      [](double value) {return std::isfinite(value);}) ||
    !std::all_of(support_steering.begin(), support_steering.end(),
      [](double value) {return std::isfinite(value) && value > 0.0;}) ||
    !std::is_sorted(support_edges.begin(), support_edges.end()) ||
    std::adjacent_find(support_edges.begin(), support_edges.end(),
      [](double lhs, double rhs) {return lhs >= rhs;}) != support_edges.end())
  {
    throw std::invalid_argument("invalid joint speed/steering support for Odom-u corrections");
  }
  reset();
}

void OdometryObserver::reset() noexcept
{
  initialized_ = false;
  turn_mode_ = false;
  turn_calm_time_s_ = 0.0;
  high_yaw_wheel_slip_active_ = false;
  high_yaw_wheel_slip_calm_time_s_ = 0.0;
  high_yaw_latent_u_mps_ = 0.0;
  high_yaw_latent_v_mps_ = 0.0;
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
  last_pre_wheel_update_speed_mps_ = 0.0;
  last_pre_wheel_update_speed_valid_ = false;
  last_wheel_raw_mps_ = 0.0;
  last_wheel_mapped_mps_ = 0.0;
  last_wheel_packet_mps_ = 0.0;
  last_turn_speed_bias_mps_ = 0.0;
  previous_actuator_steering_rad_ = 0.0;
  previous_actuator_throttle_ = 0.0;
  actuator_history_valid_ = false;
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
  const double bounded_bias = std::clamp(
    bias, -config_.turn_speed_bias_max_mps, config_.turn_speed_bias_max_mps);
  const double speed_span = config_.turn_speed_bias_full_speed_mps -
    config_.turn_speed_bias_start_speed_mps;
  double high_speed_gate = 0.0;
  if (speed_span <= 0.0) {
    high_speed_gate = speed >= config_.turn_speed_bias_full_speed_mps ? 1.0 : 0.0;
  } else {
    const double ramp = std::clamp(
      (speed - config_.turn_speed_bias_start_speed_mps) / speed_span, 0.0, 1.0);
    high_speed_gate = ramp * ramp * (3.0 - 2.0 * ramp);
  }

  // An accepted wheel update carries a repeatable speed-dependent residual in
  // hard turns. Keep this empirical correction separate from the high-speed
  // fit: practice holdouts show a positive residual at 2–4 m/s but the
  // existing correction already becomes too strong above that region.
  const double slip_entry_width =
    config_.turn_slip_speed_bias_full_speed_mps -
    config_.turn_slip_speed_bias_start_speed_mps;
  const double slip_gate_in = std::clamp(
    slip_entry_width > 0.0 ?
      (speed - config_.turn_slip_speed_bias_start_speed_mps) /
        slip_entry_width : 0.0,
    0.0, 1.0);
  const double slip_gate_in_smooth =
    slip_gate_in * slip_gate_in * (3.0 - 2.0 * slip_gate_in);
  const double slip_fade_width =
    config_.turn_slip_speed_bias_fade_end_mps -
    config_.turn_slip_speed_bias_fade_start_mps;
  const double slip_fade = std::clamp(
    slip_fade_width > 0.0 ?
      (speed - config_.turn_slip_speed_bias_fade_start_mps) /
        slip_fade_width : 0.0,
    0.0, 1.0);
  const double slip_gate_out = 1.0 -
    slip_fade * slip_fade * (3.0 - 2.0 * slip_fade);
  const double lateral_accel_proxy = speed * yaw_rate_abs;
  const double slip_correction = std::min(
    config_.turn_speed_bias_max_mps,
    config_.turn_slip_speed_bias_gain_mps_per_mps2 * std::max(
      0.0, lateral_accel_proxy -
        config_.turn_slip_speed_bias_lateral_accel_threshold_mps2));
  double lateral_accel_gate = 1.0;
  if (config_.use_turn_speed_bias_lateral_accel_gate) {
    const double gate_span =
      config_.turn_speed_bias_lateral_accel_gate_full_mps2 -
      config_.turn_speed_bias_lateral_accel_gate_start_mps2;
    const double gate_ramp = gate_span > 0.0 ? std::clamp(
      (lateral_accel_proxy -
      config_.turn_speed_bias_lateral_accel_gate_start_mps2) / gate_span,
      0.0, 1.0) :
      (lateral_accel_proxy >=
      config_.turn_speed_bias_lateral_accel_gate_full_mps2 ? 1.0 : 0.0);
    lateral_accel_gate = gate_ramp * gate_ramp * (3.0 - 2.0 * gate_ramp);
  }
  return high_speed_gate * bounded_bias * lateral_accel_gate -
    slip_gate_in_smooth * slip_gate_out * slip_correction;
}

double OdometryObserver::kinematic_base_lateral_velocity(
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
  const double velocity_at_reference = std::clamp(
    yaw_rate_radps * speed_gain,
    -config_.lateral_velocity_max_mps, config_.lateral_velocity_max_mps);
  const double reference_forward_offset = finite(
    config_.lateral_velocity_reference_forward_offset_m) ?
    config_.lateral_velocity_reference_forward_offset_m : 0.0;
  return velocity_at_reference - yaw_rate_radps * reference_forward_offset;
}

OdometryEstimate OdometryObserver::estimate(
  const OdometryObservation & observation) const noexcept
{
  OdometryEstimate output;
  output.stamp_s = observation.stamp_s;
  output.dt_s = initialized_ ? observation.stamp_s - previous_stamp_s_ : 0.0;
  output.speed_pred_mps = last_speed_pred_mps_;
  output.pre_wheel_update_speed_mps = last_pre_wheel_update_speed_mps_;
  output.pre_wheel_update_speed_valid = last_pre_wheel_update_speed_valid_;
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
  output.high_yaw_wheel_slip_active = high_yaw_wheel_slip_active_;
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
  last_pre_wheel_update_speed_valid_ = false;
  constexpr double kUnityPhysicsPacketPeriodS = 0.025;
  double steering_rate_feedback_radps = 0.0;
  double throttle_rate_feedback_per_s = 0.0;
  const bool actuator_feedback_valid = observation.actuator_feedback_valid &&
    finite(observation.steering_feedback_rad) &&
    finite(observation.throttle_feedback);
  if (actuator_feedback_valid) {
    if (actuator_history_valid_) {
      steering_rate_feedback_radps =
        (observation.steering_feedback_rad - previous_actuator_steering_rad_) /
        kUnityPhysicsPacketPeriodS;
      throttle_rate_feedback_per_s =
        (observation.throttle_feedback - previous_actuator_throttle_) /
        kUnityPhysicsPacketPeriodS;
    }
    previous_actuator_steering_rad_ = observation.steering_feedback_rad;
    previous_actuator_throttle_ = observation.throttle_feedback;
    actuator_history_valid_ = true;
  } else {
    actuator_history_valid_ = false;
  }
  if (!finite(observation.stamp_s) || !finite(observation.left_angle_rad) ||
    !finite(observation.right_angle_rad) || !finite(observation.ax_mps2) ||
    !finite(observation.ay_mps2) || !finite(observation.yaw_rate_radps) ||
    !finite(observation.yaw_rad))
  {
    high_yaw_wheel_slip_active_ = false;
    high_yaw_wheel_slip_calm_time_s_ = 0.0;
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
    high_yaw_wheel_slip_active_ = false;
    high_yaw_wheel_slip_calm_time_s_ = 0.0;
    high_yaw_latent_u_mps_ = 0.0;
    high_yaw_latent_v_mps_ = 0.0;
    body_u_mps_ = 0.0;
    body_v_mps_ = 0.0;
    speed_mps_ = 0.0;
    last_speed_pred_mps_ = 0.0;
    last_pre_wheel_update_speed_mps_ = 0.0;
    last_pre_wheel_update_speed_valid_ = false;
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
    // There is no causal interval to integrate, but the newest sample still
    // becomes the observer baseline so a source-clock reversal cannot freeze
    // all future updates.
    previous_stamp_s_ = observation.stamp_s;
    previous_left_angle_rad_ = observation.left_angle_rad;
    previous_right_angle_rad_ = observation.right_angle_rad;
    previous_yaw_rad_ = observation.yaw_rad;
    previous_yaw_rate_radps_ = observation.yaw_rate_radps;
    high_yaw_wheel_slip_active_ = false;
    high_yaw_wheel_slip_calm_time_s_ = 0.0;
    encoder_history_.clear();
    encoder_history_.push_back({
      observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
    auto output = estimate(observation);
    output.dt_s = dt_s;
    output.timing_degraded = true;
    return output;
  }

  const double previous_feature_wheel_mapped_mps = last_wheel_mapped_mps_;

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
    high_yaw_wheel_slip_active_ = false;
    high_yaw_wheel_slip_calm_time_s_ = 0.0;
    high_yaw_latent_u_mps_ = 0.0;
    high_yaw_latent_v_mps_ = 0.0;
    speed_mps_ = 0.0;
    body_u_mps_ = 0.0;
    body_v_mps_ = 0.0;
    previous_pose_body_u_mps_ = 0.0;
    previous_pose_body_v_mps_ = 0.0;
    last_speed_pred_mps_ = 0.0;
    last_pre_wheel_update_speed_mps_ = 0.0;
    last_pre_wheel_update_speed_valid_ = false;
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
    config_.wheel_speed_slew_limit_mps2;
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

  const auto wheel_speed_is_valid = [&](double predicted_speed) {
    if (!finite(wheel_mapped)) {
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
    // Permit the first positive wheel sample to establish motion. The
    // remaining checks above only reject explicit packet, launch-spin, and
    // slew anomalies.
    return true;
  };

  // A delayed bridge callback can span several nominal packets. Its average
  // wheel rate is useful only if it is also compatible with the causal
  // longitudinal prediction over that interval. The encoder bridge can
  // repeat a stale cumulative angle across a long interval while its rolling
  // and packet rates agree, so agreement between those two rates alone is
  // not sufficient to recover a moving estimate.
  double long_interval_ax_mps2 = observation.ax_mps2;
  if (observation.ax_mps2 < config_.decel_detect_ax_mps2) {
    long_interval_ax_mps2 = config_.decel_ax_scale * observation.ax_mps2 +
      config_.decel_ax_offset_mps2;
  }
  const double acceleration_reference_x_m = finite(
    config_.imu_acceleration_reference_x_m) ?
    config_.imu_acceleration_reference_x_m : 0.0;
  long_interval_ax_mps2 += observation.yaw_rate_radps *
    observation.yaw_rate_radps * acceleration_reference_x_m;
  const double long_interval_expected_average_speed_mps = std::max(
    0.0, speed_mps_ + 0.5 * long_interval_ax_mps2 * dt_s);
  const double long_interval_speed_tolerance_mps = std::max(
    config_.turn_current_packet_max_decrease_mps,
    0.5 * config_.wheel_speed_slew_limit_mps2 * dt_s);
  const bool long_interval_wheel_recovery =
    dt_s > config_.normal_packet_dt_max_s &&
    wheel_packet >= config_.wheel_freeze_speed_mps &&
    wheel_packet_mapped >= config_.wheel_freeze_speed_mps &&
    wheel_packet_mapped <= config_.wheel_burst_catchup_max_mps &&
    !wheel_burst_rejected_ && !wheel_slew_rejected &&
    !moving_encoder_dropout &&
    wheel_packet_mapped >= long_interval_expected_average_speed_mps -
    long_interval_speed_tolerance_mps &&
    wheel_packet_mapped <= long_interval_expected_average_speed_mps +
    long_interval_speed_tolerance_mps &&
    !launch_wheel_spin(speed_mps_) &&
    config_.wheel_burst_disagreement_mps > 0.0 &&
    std::abs(wheel_packet_mapped - wheel_mapped) <=
    config_.wheel_burst_disagreement_mps;

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
    high_yaw_wheel_slip_active_ = false;
    high_yaw_wheel_slip_calm_time_s_ = 0.0;
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
    high_yaw_wheel_slip_active_ = false;
    high_yaw_wheel_slip_calm_time_s_ = 0.0;
    high_yaw_latent_u_mps_ = 0.0;
    high_yaw_latent_v_mps_ = 0.0;
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
  const double imu_acceleration_reference_x = finite(
    config_.imu_acceleration_reference_x_m) ?
    config_.imu_acceleration_reference_x_m : 0.0;
  const double ax_origin = observation.ax_mps2 +
    observation.yaw_rate_radps * observation.yaw_rate_radps *
    imu_acceleration_reference_x;
  const double ay_origin = observation.ay_mps2 - yaw_alpha * imu_acceleration_reference_x;
  const double prior_body_u_mps = body_u_mps_;
  const double prior_body_v_mps = body_v_mps_;

  const bool turn_mode_at_interval_start = turn_mode_;
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
    // encoder is the trusted longitudinal measurement in turn mode.
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
    // A repeated encoder packet is a longitudinal measurement dropout, not
    // evidence that the lateral IMU acceleration is a valid vehicle-state
    // model. On the Unity car the latter contains contact/COM transients;
    // enabling it automatically during a dropout accumulated a false
    // lateral velocity of almost 1 m/s in a live MPC run. Keep the dynamic
    // option explicit so it can only be enabled by a separately validated
    // vehicle profile. The dropout path below still propagates longitudinal
    // braking from ax and uses the bounded kinematic lateral model when
    // configured.
    const bool integrate_lateral_dynamics =
      config_.integrate_lateral_acceleration_in_turn;
    // Longitudinal wheel speed is the observable with the correct scale in a
    // turn.  Use it to anchor u before and after the IMU RK2 lateral update;
    // otherwise the turn model integrates small ax bias and can report
    // 0.30 m/s while the encoders and vehicle are travelling at 0.50 m/s.
    // A repeated zero/near-zero packet makes the current-packet derivative
    // unusable, but the rolling window can still contain the real motion.
    // Recover from that ordinary dropout with the rolling rate.  The old
    // path preferred the instantaneous packet whenever it was nonzero; in the
    // simulator that selected stale low
    // rates (for example 1.55 m/s while the car was travelling about 2.7
    // m/s) and permanently put MPC behind the raceline.
    const bool window_wheel_recovery = wheel_dropout_active_ &&
      !wheel_burst_recovery_pending_ &&
      !wheel_burst_rejected_ &&
      !wheel_slew_rejected &&
      observation.ax_mps2 > config_.turn_wheel_braking_ax_mps2 &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      turn_wheel_mapped >= config_.wheel_freeze_speed_mps &&
      turn_wheel_mapped >= speed_mps_ -
      config_.turn_current_packet_max_decrease_mps;
    const bool normal_wheel_recovery = wheel_dropout_active_ &&
      !wheel_burst_rejected_ &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      (!wheel_burst_recovery_pending_ ||
      (config_.wheel_burst_disagreement_mps > 0.0 &&
      std::abs(turn_wheel_packet_mapped - turn_wheel_mapped) <=
      config_.wheel_burst_disagreement_mps)) &&
      turn_wheel_packet_mapped <= speed_mps_ +
      config_.turn_current_packet_max_increase_mps &&
      turn_wheel_packet_mapped >= speed_mps_ -
      config_.turn_current_packet_max_decrease_mps;
    const bool turn_current_packet_recovery =
      config_.allow_turn_current_packet_recovery && wheel_dropout_active_ &&
      wheel_burst_recovery_pending_ && !wheel_burst_rejected_ &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      turn_wheel_packet_mapped >= speed_mps_ &&
      turn_wheel_packet_mapped <= speed_mps_ +
      config_.turn_current_packet_max_increase_mps &&
      turn_wheel_packet_mapped >= speed_mps_ -
      config_.turn_current_packet_max_decrease_mps &&
      config_.wheel_burst_disagreement_mps > 0.0 &&
      turn_wheel_packet_mapped > turn_wheel_mapped +
      config_.wheel_burst_disagreement_mps;
    const bool packet_wheel_recovery = !window_wheel_recovery &&
      (normal_wheel_recovery || turn_current_packet_recovery);
    // After a rejected burst, a current packet that agrees with the rolling
    // rate is a coherent recovery measurement when it is moderately above
    // the stale causal speed. Keep the broader burst disagreement bound as
    // the upper limit: a rolling/current pair that is more than that above
    // the causal state is still a positive burst and must use bounded
    // catch-up instead of overwriting the state.
    const bool coherent_burst_recovery = wheel_burst_recovery_pending_ &&
      !wheel_burst_rejected_ && !wheel_slew_rejected &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      turn_wheel_mapped >= config_.wheel_freeze_speed_mps &&
      turn_wheel_mapped <= config_.wheel_burst_catchup_max_mps &&
      turn_wheel_mapped >= speed_mps_ -
      config_.turn_current_packet_max_decrease_mps &&
      turn_wheel_mapped <= speed_mps_ +
      config_.wheel_burst_disagreement_mps &&
      config_.wheel_burst_disagreement_mps > 0.0 &&
      std::abs(turn_wheel_packet_mapped - turn_wheel_mapped) <=
      config_.wheel_burst_disagreement_mps;
    const bool wheel_recovery = window_wheel_recovery ||
      packet_wheel_recovery || long_interval_wheel_recovery ||
      coherent_burst_recovery;
    const bool rolling_wheel_recovery = window_wheel_recovery ||
      coherent_burst_recovery;
    const bool burst_window_catchup =
      wheel_burst_recovery_pending_ && !wheel_recovery &&
      !launch_wheel_spin(speed_mps_) &&
      observation.ax_mps2 > config_.turn_wheel_braking_ax_mps2 &&
      config_.wheel_burst_catchup_accel_mps2 > 0.0 &&
      config_.wheel_burst_catchup_max_mps > 0.0 &&
      turn_wheel_mapped >= config_.wheel_freeze_speed_mps &&
      turn_wheel_mapped <= config_.wheel_burst_catchup_max_mps &&
      turn_wheel_mapped > body_u_mps_ +
      config_.wheel_burst_disagreement_mps;
    const bool wheel_coherent = !wheel_burst_rejected_ &&
      !launch_wheel_spin(speed_mps_) &&
      wheel_packet >= config_.wheel_freeze_speed_mps &&
      config_.wheel_burst_disagreement_mps > 0.0 &&
      !wheel_slew_rejected &&
      turn_wheel_mapped >= config_.wheel_freeze_speed_mps &&
      std::abs(turn_wheel_packet_mapped - turn_wheel_mapped) <=
      config_.wheel_burst_disagreement_mps;
    const bool wheel_ok = !wheel_burst_rejected_ && !wheel_dropout_active_ &&
      (wheel_speed_is_valid(speed_mps_) || wheel_coherent);
    last_pre_wheel_update_speed_mps_ = std::hypot(body_u_mps_, body_v_mps_);
    last_pre_wheel_update_speed_valid_ = turn_mode_at_interval_start;
    if (wheel_ok) {
      body_u_mps_ = turn_wheel_mapped;
      wheel_update_used = true;
    } else if (wheel_recovery) {
      body_u_mps_ = rolling_wheel_recovery ?
        turn_wheel_mapped : turn_wheel_packet_mapped;
      wheel_dropout_active_ = false;
      wheel_burst_recovery_pending_ = false;
      if (packet_wheel_recovery || long_interval_wheel_recovery) {
        // The rolling window still contains the repeated cumulative-angle
        // packet. Rebase it only when a current-packet recovery was actually
        // required; ordinary dropout recovery already used the window and
        // must retain it for the next delayed callback.
        encoder_history_.clear();
        encoder_history_.push_back({
          observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
      }
      wheel_update_used = true;
    } else if (burst_window_catchup) {
      // The rolling rate is useful evidence that the causal state is stale,
      // but it may still contain part of a cumulative-angle burst. Catch up
      // at a bounded rate and keep the pending flag set until a coherent
      // current packet clears the burst condition.
      body_u_mps_ = std::min(
        turn_wheel_mapped,
        body_u_mps_ + config_.wheel_burst_catchup_accel_mps2 * dt_s);
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
        imu_acceleration_reference_x;
      if (braking_ax > 0.0) {
        braking_ax = std::min(
          braking_ax, config_.wheel_dropout_positive_ax_max_mps2);
      }
      body_u_mps_ = std::max(0.0, body_u_mps_ + braking_ax * dt_s);
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
        imu_acceleration_reference_x;
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
          imu_acceleration_reference_x;
      }
      update_turn(turn_ax_origin, ay_origin, observation.yaw_rate_radps, dt_s);
      if (wheel_ok || wheel_recovery) {
        // A recovery packet is a fresh synchronized wheel measurement. The
        // dynamic update above is still needed for the lateral state, but it
        // must not overwrite the recovered longitudinal anchor with the same
        // interval's IMU acceleration. Otherwise a delayed-encoder burst
        // creates an artificial under-speed step immediately after recovery.
        body_u_mps_ = wheel_recovery ?
          (rolling_wheel_recovery ? turn_wheel_mapped : turn_wheel_packet_mapped) :
          turn_wheel_mapped;
      }
    } else {
      // The encoder is the trusted longitudinal measurement in a turn. Do
      // not integrate the lateral IMU sample into a persistent pose velocity:
      // source-time replay showed that this creates false lateral displacement
      // in the first corner. A separately identified, bounded sideslip model
      // is allowed here because it uses only causal wheel speed and yaw rate.
      body_v_mps_ = kinematic_base_lateral_velocity(
        observation.yaw_rate_radps, body_u_mps_);
    }
    if (wheel_update_used && turn_mode_at_interval_start &&
      prior_body_u_mps >= config_.turn_wheel_update_start_speed_mps &&
      observation.ax_mps2 > config_.turn_wheel_update_min_ax_mps2 &&
      config_.turn_wheel_update_innovation_threshold_mps >= 0.0 &&
      config_.turn_wheel_update_slip_beta >= 0.0)
    {
      // A one-step screen on 64 valid training phases selected this correction
      // only above 7 m/s. Propagate the prior body state with rigid-body
      // acceleration, then blend positive wheel innovations with 0.125 wheel
      // weight. Negative innovations retain the direct encoder correction.
      double turn_ax = observation.ax_mps2;
      if (turn_ax < config_.decel_detect_ax_mps2) {
        turn_ax = config_.decel_ax_scale * turn_ax +
          config_.decel_ax_offset_mps2;
      }
      turn_ax += observation.yaw_rate_radps * observation.yaw_rate_radps *
        imu_acceleration_reference_x;
      const double imu_predicted_u = std::max(
        0.0, prior_body_u_mps +
        (turn_ax + observation.yaw_rate_radps * prior_body_v_mps) * dt_s);
      const double positive_wheel_innovation = body_u_mps_ - imu_predicted_u;
      if (positive_wheel_innovation >
        config_.turn_wheel_update_innovation_threshold_mps)
      {
        const double wheel_beta = std::clamp(
          config_.turn_wheel_update_slip_beta, 0.0, 1.0);
        body_u_mps_ = (1.0 - wheel_beta) * imu_predicted_u +
          wheel_beta * body_u_mps_;
      }
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
    // During encoder dropout, continue using measured longitudinal IMU
    // acceleration. Bound positive integration to the clean-run vehicle
    // envelope so an encoder burst cannot turn an IMU transient into an
    // unbounded speed correction.
    if (wheel_dropout_active_ && ax_effective > 0.0) {
      ax_effective = std::min(
        ax_effective, config_.wheel_dropout_positive_ax_max_mps2);
    }
    speed_pred = std::max(0.0, speed_mps_ + ax_effective * dt_s);
    last_pre_wheel_update_speed_mps_ = speed_pred;
    last_pre_wheel_update_speed_valid_ = true;
    speed_mps_ = speed_pred;
    // Timing degradation is diagnostic only. Use the paired encoder sample
    // regardless of the source interval; no packet-age horizon discards it.
      // During an ordinary repeated-angle dropout, the rolling encoder rate
      // is the least burst-sensitive causal measurement. Prefer it over a
      // current-packet derivative that can be a delayed low-rate sample.
      const bool window_wheel_recovery = wheel_dropout_active_ &&
        !wheel_burst_recovery_pending_ &&
        !wheel_burst_rejected_ &&
        !wheel_slew_rejected &&
        observation.ax_mps2 > config_.turn_wheel_braking_ax_mps2 &&
        !launch_wheel_spin(speed_pred) &&
        wheel_packet >= config_.wheel_freeze_speed_mps &&
        wheel_mapped >= config_.wheel_freeze_speed_mps &&
        wheel_mapped >= speed_pred -
        config_.turn_current_packet_max_decrease_mps;
      const bool packet_wheel_recovery = wheel_dropout_active_ &&
        !wheel_burst_rejected_ &&
        !launch_wheel_spin(speed_pred) &&
        wheel_packet >= config_.wheel_freeze_speed_mps &&
        (!wheel_burst_recovery_pending_ ||
        (config_.wheel_burst_disagreement_mps > 0.0 &&
        std::abs(wheel_packet_mapped - wheel_mapped) <=
        config_.wheel_burst_disagreement_mps)) &&
        (!wheel_burst_recovery_pending_ ||
        config_.turn_current_packet_max_increase_mps <= 0.0 ||
        wheel_packet_mapped <= speed_pred +
        config_.turn_current_packet_max_increase_mps) &&
        wheel_packet_mapped >= speed_pred -
        config_.turn_current_packet_max_decrease_mps;
      const bool coherent_burst_recovery = wheel_burst_recovery_pending_ &&
        !wheel_burst_rejected_ && !wheel_slew_rejected &&
        !launch_wheel_spin(speed_pred) &&
        wheel_packet >= config_.wheel_freeze_speed_mps &&
        wheel_mapped >= config_.wheel_freeze_speed_mps &&
        wheel_mapped <= config_.wheel_burst_catchup_max_mps &&
        wheel_mapped >= speed_pred -
        config_.turn_current_packet_max_decrease_mps &&
        wheel_mapped <= speed_pred +
        config_.wheel_burst_disagreement_mps &&
        config_.wheel_burst_disagreement_mps > 0.0 &&
        std::abs(wheel_packet_mapped - wheel_mapped) <=
        config_.wheel_burst_disagreement_mps;
      const bool wheel_recovery = window_wheel_recovery ||
        (!window_wheel_recovery &&
        (packet_wheel_recovery || long_interval_wheel_recovery ||
        coherent_burst_recovery));
      const bool rolling_wheel_recovery = window_wheel_recovery ||
        coherent_burst_recovery;
      const bool burst_window_catchup =
        wheel_burst_recovery_pending_ && !wheel_recovery &&
        !launch_wheel_spin(speed_pred) &&
        observation.ax_mps2 > config_.turn_wheel_braking_ax_mps2 &&
        config_.wheel_burst_catchup_accel_mps2 > 0.0 &&
        config_.wheel_burst_catchup_max_mps > 0.0 &&
        wheel_mapped >= config_.wheel_freeze_speed_mps &&
        wheel_mapped <= config_.wheel_burst_catchup_max_mps &&
        wheel_mapped > speed_pred + config_.wheel_burst_disagreement_mps;
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
          speed_mps_ = rolling_wheel_recovery ? wheel_mapped : wheel_packet_mapped;
        wheel_dropout_active_ = false;
        wheel_burst_recovery_pending_ = false;
        if (!window_wheel_recovery) {
          // Discard the repeated-angle sample from the rolling window only
          // when the current packet, rather than the window, recovered us.
          encoder_history_.clear();
          encoder_history_.push_back({
            observation.stamp_s, observation.left_angle_rad, observation.right_angle_rad});
        }
      } else {
        speed_mps_ = speed_pred < config_.wheel_freeze_speed_mps ? wheel_mapped :
          (1.0 - config_.wheel_update_beta) * speed_pred +
          config_.wheel_update_beta * wheel_mapped;
      }
      wheel_update_used = true;
    } else if (burst_window_catchup) {
      speed_mps_ = std::min(
        wheel_mapped,
        speed_pred + config_.wheel_burst_catchup_accel_mps2 * dt_s);
    }
    body_u_mps_ = speed_mps_;
    body_v_mps_ = 0.0;
    pose_body_u_mps = body_u_mps_;
    pose_body_v_mps = body_v_mps_;
  }

  double high_yaw_published_u_mps = body_u_mps_;
  bool high_yaw_wheel_slip_replaced_u = false;
  double learned_u_published_mps = 0.0;
  bool learned_u_residual_applied = false;
  if (config_.use_high_yaw_wheel_slip_observer &&
    config_.high_yaw_wheel_slip_enter_radps > 0.0 &&
    config_.high_yaw_wheel_slip_exit_radps >= 0.0 &&
    config_.high_yaw_wheel_slip_exit_radps <
    config_.high_yaw_wheel_slip_enter_radps &&
    config_.high_yaw_wheel_slip_exit_hold_s > 0.0 &&
    config_.high_yaw_wheel_slip_innovation_mps > 0.0)
  {
    const double abs_yaw_rate = std::abs(observation.yaw_rate_radps);
    if (!high_yaw_wheel_slip_active_ &&
      abs_yaw_rate >= config_.high_yaw_wheel_slip_enter_radps)
    {
      high_yaw_wheel_slip_active_ = true;
      high_yaw_wheel_slip_calm_time_s_ = 0.0;
      high_yaw_latent_u_mps_ = prior_body_u_mps;
      high_yaw_latent_v_mps_ = prior_body_v_mps;
    }

    if (high_yaw_wheel_slip_active_) {
      high_yaw_wheel_slip_calm_time_s_ =
        abs_yaw_rate < config_.high_yaw_wheel_slip_exit_radps ?
        high_yaw_wheel_slip_calm_time_s_ + dt_s : 0.0;
      if (high_yaw_wheel_slip_calm_time_s_ >=
        config_.high_yaw_wheel_slip_exit_hold_s)
      {
        high_yaw_wheel_slip_active_ = false;
        high_yaw_wheel_slip_calm_time_s_ = 0.0;
      } else {
        // Integrate a private body-frame state using rigid-body equations.
        // Its lateral component is not promoted to published odometry.
        const double latent_du = ax_origin +
          observation.yaw_rate_radps * high_yaw_latent_v_mps_;
        const double latent_dv = ay_origin -
          observation.yaw_rate_radps * high_yaw_latent_u_mps_;
        const double latent_u_mid = high_yaw_latent_u_mps_ +
          0.5 * dt_s * latent_du;
        const double latent_v_mid = high_yaw_latent_v_mps_ +
          0.5 * dt_s * latent_dv;
        high_yaw_latent_u_mps_ += dt_s * (ax_origin +
          observation.yaw_rate_radps * latent_v_mid);
        high_yaw_latent_v_mps_ += dt_s * (ay_origin -
          observation.yaw_rate_radps * latent_u_mid);
        high_yaw_latent_u_mps_ = std::clamp(high_yaw_latent_u_mps_, -30.0, 30.0);
        high_yaw_latent_v_mps_ = std::clamp(high_yaw_latent_v_mps_, -30.0, 30.0);

        if (finite(wheel_mapped) &&
          wheel_mapped - high_yaw_latent_u_mps_ >
          config_.high_yaw_wheel_slip_innovation_mps)
        {
          high_yaw_published_u_mps = high_yaw_latent_u_mps_;
          high_yaw_wheel_slip_replaced_u = true;
        }
      }
    }
  } else {
    high_yaw_wheel_slip_active_ = false;
    high_yaw_wheel_slip_calm_time_s_ = 0.0;
  }

  if (high_yaw_wheel_slip_replaced_u) {
    // Keep the established odometry v estimator and internal wheel fusion
    // untouched; use the corrected u only for the published twist and pose.
    pose_body_u_mps = high_yaw_published_u_mps;
  }

  bool u_correction_in_supported_region = false;
  if (actuator_feedback_valid && finite(high_yaw_published_u_mps) &&
    finite(observation.steering_feedback_rad))
  {
    const auto & speed_edges = config_.u_correction_support_speed_edges_mps;
    const auto upper_edge = std::upper_bound(
      speed_edges.begin(), speed_edges.end(), high_yaw_published_u_mps);
    if (upper_edge != speed_edges.begin() && upper_edge != speed_edges.end()) {
      const std::size_t region = static_cast<std::size_t>(
        std::distance(speed_edges.begin(), upper_edge) - 1);
      u_correction_in_supported_region =
        std::abs(observation.steering_feedback_rad) <=
        config_.u_correction_support_max_abs_steering_rad[region];
    }
  }

  if (config_.use_learned_u_residual_model &&
    u_correction_in_supported_region &&
    high_yaw_published_u_mps >= config_.learned_u_residual_min_speed_mps &&
    high_yaw_published_u_mps < config_.learned_u_residual_max_speed_mps)
  {
    const double u = high_yaw_published_u_mps;
    const double v = body_v_mps_;
    const double yaw_rate = observation.yaw_rate_radps;
    const std::array<double, 22> features{
      u,
      v,
      last_wheel_mapped_mps_,
      last_wheel_packet_mps_,
      last_pre_wheel_update_speed_mps_,
      observation.ax_mps2,
      observation.ay_mps2,
      yaw_rate,
      std::abs(yaw_rate),
      yaw_rate * yaw_rate,
      u * std::abs(yaw_rate),
      u * yaw_rate * yaw_rate,
      observation.ax_mps2 * std::abs(yaw_rate),
      observation.ay_mps2 * std::abs(yaw_rate),
      last_wheel_mapped_mps_ - u,
      last_wheel_packet_mps_ - last_wheel_mapped_mps_,
      last_pre_wheel_update_speed_mps_ - u,
      dt_s,
      turn_mode_ ? 1.0 : 0.0,
      wheel_update_used ? 1.0 : 0.0,
      wheel_burst_rejected_ ? 1.0 : 0.0,
      last_turn_speed_bias_mps_};
    double residual_mps = config_.learned_u_residual_intercept_mps;
    for (std::size_t index = 0; index < features.size(); ++index) {
      residual_mps += config_.learned_u_residual_coefficients[index] * features[index];
    }
    // This fit predicts the current published-state residual of the baseline
    // observer. Do not feed its correction into that observer's latent state:
    // doing so applies the same residual again on later packets and caused
    // recursive error growth in held-out replay.
    learned_u_published_mps = high_yaw_published_u_mps + residual_mps;
    learned_u_residual_applied = true;
    pose_body_u_mps = learned_u_published_mps;
  }

  double unity_actuator_published_u_mps = learned_u_residual_applied ?
    learned_u_published_mps : high_yaw_published_u_mps;
  bool unity_actuator_model_applied = false;
  const double abs_steering = std::abs(observation.steering_feedback_rad);
  const double high_steer_u_max_speed_mps =
    observation.throttle_feedback <
    config_.high_steer_unity_actuator_u_throttle_split ?
    config_.high_steer_unity_actuator_u_max_speed_low_throttle_mps :
    config_.high_steer_unity_actuator_u_max_speed_mps;
  const bool unity_actuator_expert_active =
    config_.use_unity_actuator_u_experts && actuator_feedback_valid &&
    // The fitted source-branch residual is supported only when the current
    // encoder packet was accepted. On the r26 practice screen it overrode a
    // near-exact IMU-propagated speed during repeated-angle/burst rejection,
    // turning a 5.6 m/s state into <1 m/s. Do not treat a rejected wheel
    // measurement as usable evidence for this wheel/actuator residual.
    wheel_update_used &&
    u_correction_in_supported_region &&
    unity_actuator_published_u_mps >= config_.unity_actuator_u_min_speed_mps &&
    unity_actuator_published_u_mps < config_.unity_actuator_u_max_speed_mps;
  const bool high_steer_unity_actuator_expert_active =
    config_.use_high_steer_unity_actuator_u_expert && actuator_feedback_valid &&
    unity_actuator_published_u_mps >=
    config_.high_steer_unity_actuator_u_min_speed_mps &&
    unity_actuator_published_u_mps <= high_steer_u_max_speed_mps &&
    std::abs(observation.yaw_rate_radps) >=
    config_.high_steer_unity_actuator_u_min_abs_yaw_rate_radps &&
    abs_steering > config_.high_steer_unity_actuator_u_min_abs_steering_rad &&
    abs_steering <= config_.high_steer_unity_actuator_u_max_abs_steering_rad &&
    observation.throttle_feedback >=
    config_.high_steer_unity_actuator_u_min_throttle_feedback &&
    observation.throttle_feedback <=
    config_.high_steer_unity_actuator_u_max_throttle_feedback;
  if (unity_actuator_expert_active || high_steer_unity_actuator_expert_active) {
  const double u = unity_actuator_published_u_mps;
  const double v = body_v_mps_;
  const double yaw_rate = observation.yaw_rate_radps;
  const std::array<double, 34> model_features{
      u,
      v,
      last_wheel_mapped_mps_,
      last_wheel_packet_mps_,
      last_pre_wheel_update_speed_mps_,
      observation.ax_mps2,
      observation.ay_mps2,
      yaw_rate,
      std::abs(yaw_rate),
      yaw_rate * yaw_rate,
      u * std::abs(yaw_rate),
      u * yaw_rate * yaw_rate,
      observation.ax_mps2 * std::abs(yaw_rate),
      observation.ay_mps2 * std::abs(yaw_rate),
      last_wheel_mapped_mps_ - u,
      last_wheel_packet_mps_ - last_wheel_mapped_mps_,
      last_pre_wheel_update_speed_mps_ - u,
      dt_s,
      turn_mode_ ? 1.0 : 0.0,
      wheel_update_used ? 1.0 : 0.0,
      wheel_burst_rejected_ ? 1.0 : 0.0,
      last_turn_speed_bias_mps_,
      observation.throttle_feedback == 0.0 ? 0.0 :
        85.6 * observation.throttle_feedback / 4.0,
      observation.throttle_feedback == 0.0 ? 85.6 : 0.0,
      observation.throttle_feedback == 0.0 ? 0.0 :
        85.6 * throttle_rate_feedback_per_s / 4.0,
      observation.steering_feedback_rad,
      steering_rate_feedback_radps,
      u * u * std::tan(observation.steering_feedback_rad) / 0.324,
      std::tan(observation.steering_feedback_rad) / 0.324,
      (observation.throttle_feedback == 0.0 ? 0.0 :
        85.6 * observation.throttle_feedback / 4.0) *
        (last_wheel_mapped_mps_ / 0.059),
      (last_wheel_mapped_mps_ - previous_feature_wheel_mapped_mps) /
        kUnityPhysicsPacketPeriodS,
      (last_wheel_mapped_mps_ - previous_feature_wheel_mapped_mps) /
        kUnityPhysicsPacketPeriodS -
        (observation.ax_mps2 + 0.15532 * yaw_rate * yaw_rate),
      (observation.throttle_feedback == 0.0 ? 0.0 :
        85.6 * observation.throttle_feedback / 4.0) *
        (last_wheel_mapped_mps_ - u),
      (u * u * std::tan(observation.steering_feedback_rad) / 0.324) *
        (last_wheel_mapped_mps_ - u)};
  if (unity_actuator_expert_active)
  {
    const auto & coefficients = observation.throttle_feedback == 0.0 ?
      config_.unity_zero_throttle_coefficients :
      config_.unity_nonzero_throttle_coefficients;
    double correction_mps = observation.throttle_feedback == 0.0 ?
      config_.unity_zero_throttle_intercept_mps :
      config_.unity_nonzero_throttle_intercept_mps;
    for (std::size_t index = 0; index < model_features.size(); ++index) {
      correction_mps += coefficients[index] * model_features[index];
    }
    unity_actuator_published_u_mps += correction_mps;
    unity_actuator_model_applied = true;
    pose_body_u_mps = unity_actuator_published_u_mps;
  }

  if (high_steer_unity_actuator_expert_active)
  {
    double correction_mps = config_.high_steer_unity_actuator_u_intercept_mps;
    for (std::size_t index = 0; index < model_features.size(); ++index) {
      correction_mps += config_.high_steer_unity_actuator_u_coefficients[index] *
        model_features[index];
    }
    unity_actuator_published_u_mps += correction_mps;
    unity_actuator_model_applied = true;
    pose_body_u_mps = unity_actuator_published_u_mps;
  }
  }

  const bool physx_velocity_available = observation.physx_velocity_valid &&
    finite(observation.physx_forward_velocity_mps) &&
    finite(observation.physx_lateral_velocity_mps);
  const double physx_blend_speed_fraction = std::clamp(
    std::abs(unity_actuator_published_u_mps) /
    config_.physx_low_speed_blend_transition_mps, 0.0, 1.0);
  const double configured_physx_blend =
    config_.physx_low_speed_velocity_blend + physx_blend_speed_fraction *
    (config_.physx_velocity_blend - config_.physx_low_speed_velocity_blend);
  const double physx_blend = physx_velocity_available ? configured_physx_blend : 0.0;
  const double physx_u_blend = physx_velocity_available ?
    config_.physx_forward_velocity_blend : 0.0;
  const double physx_rear_axle_v_mps = observation.physx_lateral_velocity_mps -
    observation.yaw_rate_radps * config_.imu_acceleration_reference_x_m;
  const double fused_body_u_mps =
    (1.0 - physx_u_blend) * unity_actuator_published_u_mps +
    physx_u_blend * observation.physx_forward_velocity_mps;
  const double fused_body_v_mps =
    (1.0 - physx_blend) * body_v_mps_ + physx_blend * physx_rear_axle_v_mps;
  if (physx_u_blend > 0.0 || physx_blend > 0.0) {
    // Keep the encoder/IMU observer latent state independent: this is a
    // bounded per-packet correction, not a replacement that can recursively
    // propagate unobserved PhysX state error.
    pose_body_u_mps = fused_body_u_mps;
    pose_body_v_mps = fused_body_v_mps;
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
  if (physx_u_blend > 0.0 || physx_blend > 0.0) {
    output.body_u_mps = fused_body_u_mps;
    output.body_v_mps = fused_body_v_mps;
    output.speed_mps = std::hypot(fused_body_u_mps, fused_body_v_mps);
  } else if (high_yaw_wheel_slip_replaced_u) {
    output.body_u_mps = high_yaw_published_u_mps;
    output.speed_mps = std::hypot(high_yaw_published_u_mps, output.body_v_mps);
    output.wheel_update_used = false;
  }
  else if (learned_u_residual_applied) {
    output.body_u_mps = learned_u_published_mps;
    output.speed_mps = std::hypot(learned_u_published_mps, output.body_v_mps);
  }
  else if (unity_actuator_model_applied) {
    output.body_u_mps = unity_actuator_published_u_mps;
    output.speed_mps = std::hypot(unity_actuator_published_u_mps, output.body_v_mps);
  }
  output.timing_degraded = dt_s > config_.normal_packet_dt_max_s;
  return output;
}

}  // namespace f1tenth_localization

#include "f1tenth_localization/odometry_observer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace f1tenth_localization
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kWheelOmegaRearLeft = 2;
constexpr std::size_t kWheelOmegaRearRight = 3;
constexpr std::size_t kUnityForward = 0;
constexpr std::size_t kUnityLateralCom = 1;
constexpr std::size_t kUnityYawRate = 2;
constexpr std::size_t kUnityWheelOmegaBegin = 3;
constexpr double kUnityFixedStepS = 0.001;
constexpr double kHighSteeringInnovationFadeStartRad = 0.4;
constexpr double kUnitySteeringLimitRad = 30.0 * kPi / 180.0;

double lateral_acceleration_innovation_gain(
  const OdometryObserverConfig & config,
  const OdometryObservation & observation,
  double rear_wheel_speed_mps, double body_u_mps) noexcept
{
  if (!observation.actuator_feedback_valid ||
      observation.throttle_feedback < 0.35 ||
      std::abs(body_u_mps) < 0.5)
  {
    return config.lateral_acceleration_residual_gain;
  }
  const double physical_steering = observation.steering_feedback_rad;
  const double positive_wheel_spin_ratio =
    (rear_wheel_speed_mps - body_u_mps) / std::max(std::abs(body_u_mps), 0.5);
  if (std::abs(physical_steering) < kHighSteeringInnovationFadeStartRad ||
      positive_wheel_spin_ratio <= 1.0)
  {
    return config.lateral_acceleration_residual_gain;
  }
  const double fade = std::clamp(
    (kUnitySteeringLimitRad - std::abs(physical_steering)) /
    (kUnitySteeringLimitRad - kHighSteeringInnovationFadeStartRad),
    0.0, 1.0);
  return config.lateral_acceleration_residual_gain * fade;
}

bool all_finite(const OdometryObservation & observation) noexcept
{
  const bool scalar_state_finite = std::isfinite(observation.stamp_s) &&
         std::isfinite(observation.left_angle_rad) &&
         std::isfinite(observation.right_angle_rad) &&
         std::isfinite(observation.ax_mps2) &&
         std::isfinite(observation.ay_mps2) &&
         std::isfinite(observation.yaw_rate_radps) &&
         std::isfinite(observation.yaw_rad) &&
         std::isfinite(observation.physx_rear_axle_x_m) &&
         std::isfinite(observation.physx_rear_axle_y_m) &&
         std::isfinite(observation.roll_rad) &&
         std::isfinite(observation.pitch_rad) &&
         std::isfinite(observation.roll_rate_radps) &&
         std::isfinite(observation.pitch_rate_radps);
  if (!scalar_state_finite || !observation.physx_state_valid ||
      !observation.physx_pose_valid) return false;
  return std::all_of(
           observation.physx_state.begin(), observation.physx_state.end(),
           [](double value) {return std::isfinite(value);}) &&
         std::all_of(observation.physx_wheel_omega_radps.begin(),
           observation.physx_wheel_omega_radps.end(),
           [](double value) {return std::isfinite(value);}) &&
         std::all_of(observation.physx_wheel_rotation_angle_rad.begin(),
           observation.physx_wheel_rotation_angle_rad.end(),
           [](double value) {return std::isfinite(value);}) &&
         std::all_of(
           observation.physx_wheel_encoder_angle_prediction_rad.begin(),
           observation.physx_wheel_encoder_angle_prediction_rad.end(),
           [](double value) {return std::isfinite(value);});
}

double wrap_angle(double angle) noexcept
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

void integrate_pose(
  double dt_s, double previous_yaw, double next_yaw,
  double previous_u, double next_u, double previous_v, double next_v,
  double & x_m, double & y_m) noexcept
{
  const double yaw_mid = wrap_angle(
    previous_yaw + 0.5 * wrap_angle(next_yaw - previous_yaw));
  const double u_mid = 0.5 * (previous_u + next_u);
  const double v_mid = 0.5 * (previous_v + next_v);
  x_m += (u_mid * std::cos(yaw_mid) - v_mid * std::sin(yaw_mid)) * dt_s;
  y_m += (u_mid * std::sin(yaw_mid) + v_mid * std::cos(yaw_mid)) * dt_s;
}

}  // namespace

OdometryObserverConfig deployment_observer_config()
{
  return {};
}

OdometryObserver::OdometryObserver(OdometryObserverConfig config)
: config_(config)
{
  if (!(config_.wheel_radius_m > 0.0) ||
      !(config_.encoder_ticks_per_revolution > 0.0) ||
      !(config_.maximum_packet_dt_s > 0.0) ||
      !std::isfinite(config_.wheel_radius_m) ||
      !std::isfinite(config_.encoder_ticks_per_revolution) ||
      !std::isfinite(config_.imu_reference_forward_m) ||
      !std::isfinite(config_.maximum_packet_dt_s) ||
      !std::isfinite(config_.lateral_acceleration_residual_gain) ||
      config_.lateral_acceleration_residual_gain < 0.0 ||
      config_.lateral_acceleration_residual_gain > 1.0)
  {
    throw std::invalid_argument("invalid source-state odometry configuration");
  }
}

void OdometryObserver::reset() noexcept
{
  initialized_ = false;
  previous_stamp_s_ = 0.0;
  previous_yaw_rad_ = 0.0;
  previous_encoder_left_angle_rad_ = 0.0;
  previous_encoder_right_angle_rad_ = 0.0;
  previous_physx_wheel_encoder_angle_prediction_rad_.fill(0.0);
  physx_pose_origin_x_m_ = 0.0;
  physx_pose_origin_y_m_ = 0.0;
  body_u_mps_ = 0.0;
  body_v_com_mps_ = 0.0;
  body_v_mps_ = 0.0;
  horizontal_u_mps_ = 0.0;
  horizontal_v_mps_ = 0.0;
  x_m_ = 0.0;
  y_m_ = 0.0;
  last_ = {};
}

OdometryEstimate OdometryObserver::current_estimate() const noexcept
{
  return last_;
}

OdometryEstimate OdometryObserver::update(
  const OdometryObservation & observation)
{
  if (!all_finite(observation)) {
    return current_estimate();
  }

  const double sr = std::sin(observation.roll_rad);
  const double cr = std::cos(observation.roll_rad);
  const double sp = std::sin(observation.pitch_rad);
  const double cp = std::cos(observation.pitch_rad);
  if (std::abs(cp * cr) < 1.0e-8) return current_estimate();
  const auto horizontal_velocity = [&](double u, double v_com) {
      // Source plant heave is WORLD vertical speed, while u/v are body-frame
      // COM components. Recover body w, transport velocity to the rear axle,
      // then project with R_y(pitch) R_x(roll). The final yaw rotation remains
      // measured IMU yaw in integrate_pose; no gravity or r*u is added twice.
      const double w_com =
        (observation.physx_state[UNITY_SOURCE_BODY_HEIGHT_RATE] + sp * u -
        cp * sr * v_com) / (cp * cr);
      const double v_rear = v_com -
        observation.yaw_rate_radps * config_.imu_reference_forward_m;
      const double w_rear = w_com +
        observation.pitch_rate_radps * config_.imu_reference_forward_m;
      return std::pair<double, double>{
        cp * u + sp * (sr * v_rear + cr * w_rear),
        cr * v_rear - sr * w_rear};
    };

  if (!initialized_) {
    initialized_ = true;
    previous_stamp_s_ = observation.stamp_s;
    previous_yaw_rad_ = observation.yaw_rad;
    body_u_mps_ = observation.physx_forward_velocity_mps;
    body_v_com_mps_ = observation.physx_lateral_velocity_com_mps;
    body_v_mps_ = body_v_com_mps_ -
      observation.yaw_rate_radps * config_.imu_reference_forward_m;
    const auto horizontal = horizontal_velocity(body_u_mps_, body_v_com_mps_);
    horizontal_u_mps_ = horizontal.first;
    horizontal_v_mps_ = horizontal.second;
    physx_pose_origin_x_m_ = observation.physx_rear_axle_x_m;
    physx_pose_origin_y_m_ = observation.physx_rear_axle_y_m;
    x_m_ = 0.0;
    y_m_ = 0.0;
    const double lateral_accel_gain = lateral_acceleration_innovation_gain(
      config_, observation,
      config_.wheel_radius_m * 0.5 *
      (observation.physx_wheel_omega_radps[kWheelOmegaRearLeft] +
      observation.physx_wheel_omega_radps[kWheelOmegaRearRight]),
      body_u_mps_);
    previous_encoder_left_angle_rad_ = observation.left_angle_rad;
    previous_encoder_right_angle_rad_ = observation.right_angle_rad;
    previous_physx_wheel_encoder_angle_prediction_rad_ =
      observation.physx_wheel_encoder_angle_prediction_rad;
    last_ = {};
    last_.stamp_s = observation.stamp_s;
    last_.x_m = x_m_;
    last_.y_m = y_m_;
    last_.physx_model_x_m = 0.0;
    last_.physx_model_y_m = 0.0;
    last_.yaw_rad = previous_yaw_rad_;
    last_.imu_yaw_rad = observation.yaw_rad;
    last_.yaw_rate_radps = observation.yaw_rate_radps;
    last_.body_u_mps = body_u_mps_;
    last_.body_v_com_mps = body_v_com_mps_;
    last_.body_v_mps = body_v_mps_;
    last_.speed_mps = std::hypot(body_u_mps_, body_v_mps_);
    last_.left_angle_rad = observation.left_angle_rad;
    last_.right_angle_rad = observation.right_angle_rad;
    last_.ax_mps2 = observation.ax_mps2;
    last_.ay_mps2 = observation.ay_mps2;
    last_.wheel_omega_radps = observation.physx_wheel_omega_radps;
    last_.longitudinal_slip = observation.physx_longitudinal_slip;
    last_.lateral_slip = observation.physx_lateral_slip;
    last_.normal_load_n = observation.physx_normal_load_n;
    last_.unity_state = observation.physx_state;
    last_.unity_state[9] = observation.roll_rad;
    last_.unity_state[10] = observation.roll_rate_radps;
    last_.unity_state[11] = observation.pitch_rad;
    last_.unity_state[12] = observation.pitch_rate_radps;
    // The runtime has already converted the one-fixed-step-old feedback into
    // the current post-Steer rack state. Keep that causal plant state instead
    // of publishing the raw feedback value as if it were current rack angle.
    last_.roll_rad = observation.roll_rad;
    last_.roll_rate_radps = observation.roll_rate_radps;
    last_.pitch_rad = observation.pitch_rad;
    last_.pitch_rate_radps = observation.pitch_rate_radps;
    last_.lateral_acceleration_residual_gain = lateral_accel_gain;
    last_.valid = observation.physx_state_valid;
    return last_;
  }

  const double measured_dt_s = observation.stamp_s - previous_stamp_s_;
  if (!(measured_dt_s > 0.0) || !std::isfinite(measured_dt_s)) {
    return current_estimate();
  }
  const double dt_s = std::min(measured_dt_s, config_.maximum_packet_dt_s);

  const double left_delta = observation.left_angle_rad -
    previous_encoder_left_angle_rad_;
  const double right_delta = observation.right_angle_rad -
    previous_encoder_right_angle_rad_;
  const bool encoder_epoch = std::abs(left_delta) > 50.0 ||
    std::abs(right_delta) > 50.0;
  const bool wheel_measurement_valid = !encoder_epoch && dt_s > 0.0;
  const double left_omega_average = wheel_measurement_valid ? left_delta / dt_s :
    observation.physx_wheel_omega_radps[kWheelOmegaRearLeft];
  const double right_omega_average = wheel_measurement_valid ? right_delta / dt_s :
    observation.physx_wheel_omega_radps[kWheelOmegaRearRight];
  const double encoder_quantum_rad =
    2.0 * kPi / config_.encoder_ticks_per_revolution;
  const auto integrated_wheel_innovation = [&](std::size_t wheel,
      double measured_delta) {
      const double model_delta =
        observation.physx_wheel_encoder_angle_prediction_rad[wheel] -
        previous_physx_wheel_encoder_angle_prediction_rad_[wheel];
      const double residual = measured_delta - model_delta;
      const double resolvable_residual = std::copysign(
        std::max(0.0, std::abs(residual) - encoder_quantum_rad), residual);
      return std::pair<double, double>{residual, 2.0 * resolvable_residual / dt_s};
    };
  const auto left_innovation = integrated_wheel_innovation(
    kWheelOmegaRearLeft, left_delta);
  const auto right_innovation = integrated_wheel_innovation(
    kWheelOmegaRearRight, right_delta);
  const double left_omega_correction = wheel_measurement_valid ?
    left_innovation.second : 0.0;
  const double right_omega_correction = wheel_measurement_valid ?
    right_innovation.second : 0.0;
  const double left_omega = observation.physx_wheel_omega_radps[
    kWheelOmegaRearLeft] + left_omega_correction;
  const double right_omega = observation.physx_wheel_omega_radps[
    kWheelOmegaRearRight] + right_omega_correction;
  const double encoder_rear_axle_speed = config_.wheel_radius_m * 0.5 *
    (left_omega_average + right_omega_average);
  previous_encoder_left_angle_rad_ = observation.left_angle_rad;
  previous_encoder_right_angle_rad_ = observation.right_angle_rad;
  previous_physx_wheel_encoder_angle_prediction_rad_ =
    observation.physx_wheel_encoder_angle_prediction_rad;

  const double prior_u = body_u_mps_;
  const double prior_v_com = body_v_com_mps_;
  const double prior_yaw = previous_yaw_rad_;
  const double predicted_u = observation.physx_forward_velocity_mps;
  const double predicted_v_com = observation.physx_lateral_velocity_com_mps;
  const double predicted_ax = observation.physx_longitudinal_acceleration_mps2;
  const double predicted_ay = observation.physx_lateral_acceleration_mps2;

  // UnityPlantState and the Unity IMU source define these accelerations as
  // the latest 1 ms FixedUpdate sample. The native plant has already advanced
  // all 25 fixed steps to this sensor epoch, so this same-epoch innovation may
  // correct only the matching final physics step, not the full 25 ms packet
  // interval. Both accelerations are inertial acceleration in body axes;
  // only the measured/model yaw-rate difference adds a velocity-frame
  // transport correction. Adding nominal r*v or r*u here double-counts it.
  const double model_yaw_rate = observation.physx_yaw_rate_radps;
  const double yaw_rate_residual =
    observation.yaw_rate_radps - model_yaw_rate;
  const double lateral_accel_gain = lateral_acceleration_innovation_gain(
    config_, observation,
    config_.wheel_radius_m * 0.5 *
    (left_omega_average + right_omega_average), predicted_u);
  const double u_acceleration_residual = observation.ax_mps2 - predicted_ax +
    yaw_rate_residual * 0.5 * (prior_v_com + predicted_v_com);
  const double v_acceleration_residual =
    lateral_accel_gain * (observation.ay_mps2 - predicted_ay) -
    yaw_rate_residual * 0.5 * (prior_u + predicted_u);
  const double next_u = predicted_u + u_acceleration_residual * kUnityFixedStepS;
  const double next_v_com = predicted_v_com +
    v_acceleration_residual * kUnityFixedStepS;
  // Unity's IMU publishes the vehicle Transform quaternion at this sensor
  // epoch. Use that measured attitude for pose integration instead of
  // integrating gyro rate and accumulating its packet-to-packet bias. Keep
  // gyro yaw rate as the instantaneous angular-velocity measurement used by
  // the plant and velocity-frame corrections.
  const double next_yaw = wrap_angle(observation.yaw_rad);
  const double next_v = next_v_com -
    observation.yaw_rate_radps * config_.imu_reference_forward_m;

  /* The public PhysX source plant supplies the motion predictor, but its raw
   * world pose is not the best measured trajectory: on the held-out practice
   * capture, its aligned position RMSE was 0.194 m versus 0.118 m for the
   * existing estimator. Integrate the sensor-corrected body velocity instead;
   * retain the independent millisecond PhysX pose below for online diagnostics. */
  const auto horizontal = horizontal_velocity(next_u, next_v_com);
  integrate_pose(dt_s, prior_yaw, next_yaw,
    horizontal_u_mps_, horizontal.first, horizontal_v_mps_, horizontal.second,
    x_m_, y_m_);
  horizontal_u_mps_ = horizontal.first;
  horizontal_v_mps_ = horizontal.second;

  body_u_mps_ = next_u;
  body_v_com_mps_ = next_v_com;
  body_v_mps_ = next_v;
  previous_stamp_s_ = observation.stamp_s;
  previous_yaw_rad_ = next_yaw;
  OdometryEstimate output;
  output.stamp_s = observation.stamp_s;
  output.dt_s = dt_s;
  output.speed_pred_mps = predicted_u;
  output.speed_mps = std::hypot(next_u, next_v);
  output.body_u_mps = next_u;
  output.body_v_mps = next_v;
  output.body_v_com_mps = next_v_com;
  output.x_m = x_m_;
  output.y_m = y_m_;
  output.physx_model_x_m =
    observation.physx_rear_axle_x_m - physx_pose_origin_x_m_;
  output.physx_model_y_m =
    observation.physx_rear_axle_y_m - physx_pose_origin_y_m_;
  output.yaw_rad = next_yaw;
  output.left_angle_rad = observation.left_angle_rad;
  output.right_angle_rad = observation.right_angle_rad;
  output.imu_yaw_rad = observation.yaw_rad;
  output.encoder_rear_axle_speed_mps = encoder_rear_axle_speed;
  output.ax_mps2 = observation.ax_mps2;
  output.ay_mps2 = observation.ay_mps2;
  output.yaw_rate_radps = observation.yaw_rate_radps;
  output.wheel_omega_radps = observation.physx_wheel_omega_radps;
  output.wheel_omega_radps[kWheelOmegaRearLeft] = left_omega;
  output.wheel_omega_radps[kWheelOmegaRearRight] = right_omega;
  output.encoder_model_rotation_residual_rad = {
    left_innovation.first, right_innovation.first};
  output.wheel_omega_correction_radps = {
    left_omega_correction, right_omega_correction};
  output.encoder_wheel_omega_average_radps = {
    left_omega_average, right_omega_average};
  output.longitudinal_slip = observation.physx_longitudinal_slip;
  output.lateral_slip = observation.physx_lateral_slip;
  output.normal_load_n = observation.physx_normal_load_n;
  output.unity_state = observation.physx_state;
  output.unity_state[kUnityForward] = next_u;
  output.unity_state[kUnityLateralCom] = next_v_com;
  output.unity_state[kUnityYawRate] = observation.yaw_rate_radps;
  output.unity_state[kUnityWheelOmegaBegin + kWheelOmegaRearLeft] = left_omega;
  output.unity_state[kUnityWheelOmegaBegin + kWheelOmegaRearRight] = right_omega;
  output.unity_state[9] = observation.roll_rad;
  output.unity_state[10] = observation.roll_rate_radps;
  output.unity_state[11] = observation.pitch_rad;
  output.unity_state[12] = observation.pitch_rate_radps;
  // Preserve the runtime's endpoint-reconstructed rack state in unity_state.
  // steering_feedback_rad remains separately available as the raw sensor.
  output.roll_rad = observation.roll_rad;
  output.roll_rate_radps = observation.roll_rate_radps;
  output.pitch_rad = observation.pitch_rad;
  output.pitch_rate_radps = observation.pitch_rate_radps;
  output.longitudinal_acceleration_mps2 = observation.ax_mps2;
  output.lateral_acceleration_mps2 = observation.ay_mps2;
  output.lateral_acceleration_residual_gain = lateral_accel_gain;
  output.valid = observation.physx_state_valid;
  last_ = output;
  return output;
}

}  // namespace f1tenth_localization

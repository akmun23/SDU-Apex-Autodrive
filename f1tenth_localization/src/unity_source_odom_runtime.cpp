#include "f1tenth_localization/physx_odom_runtime.hpp"

#include "unity_source_plant_c.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace f1tenth_localization
{
namespace
{

constexpr double kFixedStepS = 0.001;
constexpr unsigned int kStepsPerPacket = 25;
constexpr double kPacketPeriodS = 0.025;
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr unsigned int kInitialRestSettleSteps = 1000;
constexpr double kUnitySteeringRateRadps = 183.346 * kPi / 180.0;
// Begin treating the prior innovation as stale once feedback has moved by
// half the source-configured maximum steering travel in one sensor epoch.
constexpr double kYawForecastSteeringTransitionRad =
  0.5 * kUnitySteeringRateRadps * kPacketPeriodS;

void copy_plant_state(
  const f1tenth_mpc::UnitySourcePlantState & input,
  std::array<double, UNITY_SOURCE_PLANT_STATE_SIZE> * output)
{
  (*output)[UNITY_SOURCE_U] = input.u_mps;
  (*output)[UNITY_SOURCE_V] = input.v_mps;
  (*output)[UNITY_SOURCE_YAW_RATE] = input.yaw_rate_radps;
  std::copy(input.wheel_omega_radps.begin(), input.wheel_omega_radps.end(),
    output->begin() + UNITY_SOURCE_WHEEL_OMEGA_FL);
  (*output)[UNITY_SOURCE_BODY_HEIGHT] = input.body_height_m;
  (*output)[UNITY_SOURCE_BODY_HEIGHT_RATE] = input.body_height_rate_mps;
  (*output)[UNITY_SOURCE_ROLL] = input.roll_rad;
  (*output)[UNITY_SOURCE_ROLL_RATE] = input.roll_rate_radps;
  (*output)[UNITY_SOURCE_PITCH] = input.pitch_rad;
  (*output)[UNITY_SOURCE_PITCH_RATE] = input.pitch_rate_radps;
  (*output)[UNITY_SOURCE_PHYSICAL_STEERING] = input.physical_steering_rad;
  std::copy(input.suspension_jounce_m.begin(), input.suspension_jounce_m.end(),
    output->begin() + UNITY_SOURCE_SUSPENSION_JOUNCE_FL);
  std::copy(input.low_forward_speed_time_s.begin(),
    input.low_forward_speed_time_s.end(),
    output->begin() + UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL);
  std::copy(input.low_side_speed_time_s.begin(),
    input.low_side_speed_time_s.end(),
    output->begin() + UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL);
}

void integrate_rear_axle_pose(
  const f1tenth_mpc::UnitySourcePlantState & before,
  const f1tenth_mpc::UnitySourcePlantState & after,
  double rear_axle_from_com_m, double dt_s,
  double * yaw_rad, double * x_m, double * y_m)
{
  const double yaw_rate_mid = 0.5 * (before.yaw_rate_radps + after.yaw_rate_radps);
  const double next_yaw = *yaw_rad + yaw_rate_mid * dt_s;
  const double yaw_mid = 0.5 * (*yaw_rad + next_yaw);
  const double u_mid = 0.5 * (before.u_mps + after.u_mps);
  const double v_rear_before =
    before.v_mps - before.yaw_rate_radps * rear_axle_from_com_m;
  const double v_rear_after =
    after.v_mps - after.yaw_rate_radps * rear_axle_from_com_m;
  const double v_rear_mid = 0.5 * (v_rear_before + v_rear_after);
  *x_m += dt_s * (u_mid * std::cos(yaw_mid) - v_rear_mid * std::sin(yaw_mid));
  *y_m += dt_s * (u_mid * std::sin(yaw_mid) + v_rear_mid * std::cos(yaw_mid));
  *yaw_rad = next_yaw;
}

}  // namespace

SourceDerivedOdomRuntime::SourceDerivedOdomRuntime(double rear_axle_from_com_m)
: rear_axle_from_com_m_(rear_axle_from_com_m)
{
  if (!std::isfinite(rear_axle_from_com_m_) || rear_axle_from_com_m_ < 0.0) {
    throw std::invalid_argument("rear axle offset must be finite and nonnegative");
  }

  // Begin with the source-model reset state and settle suspension/jounce at
  // the same 1 ms cadence before the first sensor epoch is consumed.
  const f1tenth_mpc::UnitySourcePlantInput reset_input{0.0, 0.0};
  f1tenth_mpc::UnitySourcePlantOutput settled{};
  for (unsigned int step = 0; step < kInitialRestSettleSteps; ++step) {
    if (!f1tenth_mpc::unity_source_plant_step_1ms(
        plant_state_, reset_input, &settled))
    {
      throw std::runtime_error("source-derived odometry plant failed to settle");
    }
    plant_state_ = settled.state;
  }
}

PredictedPlantState SourceDerivedOdomRuntime::advance(
  double steering_command_norm, double throttle_command_norm,
  bool steering_feedback_valid, double steering_feedback_rad)
{
  if (!std::isfinite(steering_command_norm) ||
    !std::isfinite(throttle_command_norm) ||
    (steering_feedback_valid && !std::isfinite(steering_feedback_rad)))
  {
    throw std::runtime_error("non-finite input sent to source-derived odometry plant");
  }

  const double applied_steering = delayed_steering_commands_norm_[0];
  const double applied_throttle = delayed_throttle_commands_norm_[0];
  delayed_steering_commands_norm_[0] = delayed_steering_commands_norm_[1];
  delayed_steering_commands_norm_[1] = std::clamp(steering_command_norm, -1.0, 1.0);
  delayed_throttle_commands_norm_[0] = delayed_throttle_commands_norm_[1];
  delayed_throttle_commands_norm_[1] = std::clamp(throttle_command_norm, -1.0, 1.0);

  if (steering_feedback_valid) {
    plant_state_.physical_steering_rad = std::clamp(
      steering_feedback_rad, -UNITY_SOURCE_MAX_STEERING_RAD,
      UNITY_SOURCE_MAX_STEERING_RAD);
  }
  const double steering_feedback_delta = steering_feedback_valid &&
    have_steering_feedback_ ?
    steering_feedback_rad - last_steering_feedback_rad_ : 0.0;

  const f1tenth_mpc::UnitySourcePlantInput input{
    f1tenth_mpc::unity_steering_command_normalized_to_physical_rad(
      applied_steering, UNITY_SOURCE_MAX_STEERING_RAD),
    applied_throttle};
  PredictedPlantState result{};
  f1tenth_mpc::UnitySourcePlantOutput physics_output{};

  for (unsigned int tick = 0; tick < kStepsPerPacket; ++tick) {
    for (std::size_t wheel = 0; wheel < wheel_encoder_prediction_.size(); ++wheel) {
      const float rpm = static_cast<float>(
        plant_state_.wheel_omega_radps[wheel] * (60.0 / (2.0 * kPi)));
      wheel_encoder_prediction_[wheel].sample_pre_solve_rpm(
        rpm, static_cast<float>(kFixedStepS));
    }

    const auto previous_state = plant_state_;
    const bool final_tick = tick + 1 == kStepsPerPacket;
    const bool stepped = final_tick
      ? f1tenth_mpc::unity_source_plant_step_1ms(
          plant_state_, input, &physics_output)
      : f1tenth_mpc::unity_source_plant_step_1ms_minimal(
          plant_state_, input, &physics_output);
    if (!stepped)
    {
      throw std::runtime_error("source-derived odometry plant rejected its state");
    }
    plant_state_ = physics_output.state;
    integrate_rear_axle_pose(previous_state, plant_state_, rear_axle_from_com_m_,
      kFixedStepS, &yaw_rad_, &rear_axle_x_m_, &rear_axle_y_m_);
  }

  copy_plant_state(plant_state_, &result.state);
  result.raw_yaw_rate_prediction_radps = plant_state_.yaw_rate_radps;
  result.corrected_yaw_rate_prediction_radps =
    result.raw_yaw_rate_prediction_radps;
  result.yaw_rate_prediction_mode = 0;
  last_raw_yaw_rate_prediction_radps_ = result.raw_yaw_rate_prediction_radps;
  have_raw_yaw_rate_prediction_ = true;
  if (have_previous_yaw_rate_prediction_residual_) {
    const bool steering_transition = steering_feedback_valid &&
      have_steering_feedback_ &&
      std::abs(steering_feedback_delta) >
      kYawForecastSteeringTransitionRad;
    const bool continuing_steering_ramp = steering_transition &&
      have_steering_feedback_delta_ &&
      std::abs(previous_steering_feedback_delta_rad_) >
      kYawForecastSteeringTransitionRad &&
      steering_feedback_delta * previous_steering_feedback_delta_rad_ > 0.0;
    const bool settling_after_steering_transition =
      steering_feedback_valid && have_steering_feedback_delta_ &&
      !steering_transition &&
      std::abs(previous_steering_feedback_delta_rad_) >
      kYawForecastSteeringTransitionRad;

    if (continuing_steering_ramp && have_yaw_rate_measurement_ &&
      have_previous_yaw_rate_measurement_)
    {
      // In a sustained physical-steering ramp, carry the measured gyro-rate
      // slope for one 25 ms step. The source plant remains the raw comparator;
      // this causal trend is only the separately published yaw forecast.
      result.corrected_yaw_rate_prediction_radps =
        last_yaw_rate_measurement_radps_ +
        (last_yaw_rate_measurement_radps_ -
        previous_yaw_rate_measurement_radps_);
      result.yaw_rate_prediction_mode = 2;
    } else if (steering_transition || settling_after_steering_transition) {
      // A residual learned in the preceding steering regime is stale during a
      // reversal or the first packet after it. Keep the source-plant forecast.
      result.yaw_rate_prediction_mode = 0;
    } else {
      // In a settled steering regime, cancel only the previous packet's
      // measured source-model residual. Never inject this correction into the
      // state propagated by the plant.
      result.yaw_rate_prediction_correction_radps =
        previous_yaw_rate_prediction_residual_radps_;
      result.corrected_yaw_rate_prediction_radps -=
        result.yaw_rate_prediction_correction_radps;
      result.yaw_rate_prediction_mode = 1;
    }
    result.yaw_rate_prediction_correction_radps =
      result.raw_yaw_rate_prediction_radps -
      result.corrected_yaw_rate_prediction_radps;
  }
  if (steering_feedback_valid) {
    if (have_steering_feedback_) {
      previous_steering_feedback_delta_rad_ = steering_feedback_delta;
      have_steering_feedback_delta_ = true;
    } else {
      previous_steering_feedback_delta_rad_ = 0.0;
      have_steering_feedback_delta_ = false;
    }
    last_steering_feedback_rad_ = steering_feedback_rad;
    have_steering_feedback_ = true;
  } else {
    have_steering_feedback_ = false;
    have_steering_feedback_delta_ = false;
  }
  result.rear_axle_x_m = rear_axle_x_m_;
  result.rear_axle_y_m = rear_axle_y_m_;
  result.wheel_rotation_angle_rad = plant_state_.wheel_rotation_angle_rad;
  for (std::size_t wheel = 0; wheel < wheel_encoder_prediction_.size(); ++wheel) {
    result.wheel_encoder_angle_prediction_rad[wheel] =
      wheel_encoder_prediction_[wheel].angle_rad;
  }
  result.longitudinal_slip = physics_output.longitudinal_slip;
  result.lateral_slip = physics_output.lateral_slip;
  result.normal_load_n = physics_output.normal_load_n;
  result.suspension_force_n = physics_output.suspension_force_n;
  result.suspension_jounce_rate_mps = physics_output.suspension_jounce_rate_mps;
  result.tire_longitudinal_force_n = physics_output.tire_longitudinal_force_n;
  result.tire_lateral_force_n = physics_output.tire_lateral_force_n;
  result.longitudinal_acceleration_mps2 =
    physics_output.longitudinal_acceleration_mps2;
  result.lateral_acceleration_mps2 = physics_output.lateral_acceleration_mps2;
  result.yaw_acceleration_radps2 = physics_output.yaw_acceleration_radps2;
  result.suspension_history_initialized = plant_state_.suspension_history_initialized;
  result.delayed_steering_commands_normalized = delayed_steering_commands_norm_;
  result.delayed_throttle_commands_normalized = delayed_throttle_commands_norm_;
  return result;
}

double SourceDerivedOdomRuntime::yaw_rate_prediction_residual_radps() const
{
  return previous_yaw_rate_prediction_residual_radps_;
}

bool SourceDerivedOdomRuntime::yaw_rate_prediction_residual_valid() const
{
  return yaw_rate_prediction_residual_valid_;
}

void SourceDerivedOdomRuntime::correct(
  double body_u_mps, double body_v_com_mps, double yaw_rate_radps,
  double yaw_rad, double roll_rad, double pitch_rad,
  double body_vertical_velocity_world_mps,
  double roll_rate_radps, double pitch_rate_radps,
  const std::array<double, 4> & wheel_omega_radps)
{
  const std::array<double, 9> scalar_values{
    body_u_mps, body_v_com_mps, yaw_rate_radps, yaw_rad, roll_rad,
    pitch_rad, body_vertical_velocity_world_mps, roll_rate_radps,
    pitch_rate_radps};
  if (!std::all_of(scalar_values.begin(), scalar_values.end(),
      [](double value) {return std::isfinite(value);}) ||
    !std::all_of(wheel_omega_radps.begin(), wheel_omega_radps.end(),
      [](double value) {return std::isfinite(value);}))
  {
    throw std::runtime_error("invalid sensor correction for source-derived odometry plant");
  }

  if (have_raw_yaw_rate_prediction_) {
    previous_yaw_rate_prediction_residual_radps_ =
      last_raw_yaw_rate_prediction_radps_ - yaw_rate_radps;
    have_previous_yaw_rate_prediction_residual_ = true;
    yaw_rate_prediction_residual_valid_ = true;
    have_raw_yaw_rate_prediction_ = false;
  }

  if (have_yaw_rate_measurement_) {
    previous_yaw_rate_measurement_radps_ = last_yaw_rate_measurement_radps_;
    have_previous_yaw_rate_measurement_ = true;
  }
  last_yaw_rate_measurement_radps_ = yaw_rate_radps;
  have_yaw_rate_measurement_ = true;

  plant_state_.u_mps = body_u_mps;
  plant_state_.v_mps = body_v_com_mps;
  plant_state_.yaw_rate_radps = yaw_rate_radps;
  plant_state_.roll_rad = roll_rad;
  plant_state_.pitch_rad = pitch_rad;
  plant_state_.body_height_rate_mps = body_vertical_velocity_world_mps;
  plant_state_.roll_rate_radps = roll_rate_radps;
  plant_state_.pitch_rate_radps = pitch_rate_radps;
  plant_state_.wheel_omega_radps = wheel_omega_radps;
  yaw_rad_ = yaw_rad;
}

}  // namespace f1tenth_localization

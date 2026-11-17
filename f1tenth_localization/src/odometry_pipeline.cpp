#include "f1tenth_localization/odometry_pipeline.hpp"

#include <algorithm>
#include <chrono>

namespace f1tenth_localization
{
namespace
{

double microseconds_since(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::micro>(
    std::chrono::steady_clock::now() - start).count();
}

}  // namespace

OdometryPipeline::OdometryPipeline(const OdometryObserverConfig & config)
: observer_(config), runtime_(config.imu_reference_forward_m)
{
}

OdometryPipelineStep OdometryPipeline::step(
  const SensorPacket & packet,
  double steering_command_norm, double throttle_command_norm,
  double steering_feedback_rad, double throttle_feedback_norm)
{
  stamp_s_ = started_ ? stamp_s_ + kPacketPeriodS : 0.0;
  started_ = true;

  OdometryPipelineStep result;
  auto start = std::chrono::steady_clock::now();
  plant_ = runtime_.advance(
    steering_command_norm, throttle_command_norm, true, steering_feedback_rad,
    true, throttle_feedback_norm);
  result.plant_advance_us = microseconds_since(start);
  result.raw_plant = plant_;

  OdometryObservation observation;
  observation.stamp_s = stamp_s_;
  observation.left_angle_rad = packet.left_angle_rad;
  observation.right_angle_rad = packet.right_angle_rad;
  observation.ax_mps2 = packet.ax_mps2;
  observation.ay_mps2 = packet.ay_mps2;
  observation.yaw_rate_radps = packet.yaw_rate_radps;
  observation.yaw_rad = packet.imu_yaw_rad;  // Only the initial heading uses orientation.
  observation.roll_rad = packet.roll_rad;
  observation.pitch_rad = packet.pitch_rad;
  observation.roll_rate_radps = packet.roll_rate_radps;
  observation.pitch_rate_radps = packet.pitch_rate_radps;
  observation.steering_feedback_rad = steering_feedback_rad;
  observation.throttle_feedback = throttle_feedback_norm;
  observation.actuator_feedback_valid = true;
  observation.steering_command_normalized = steering_command_norm;
  observation.throttle_command_normalized = throttle_command_norm;
  observation.actuator_command_valid = true;
  observation.physx_forward_velocity_mps = plant_.state[0];
  observation.physx_lateral_velocity_com_mps = plant_.state[1];
  observation.physx_yaw_rate_radps = plant_.state[2];
  observation.physx_longitudinal_acceleration_mps2 = plant_.longitudinal_acceleration_mps2;
  observation.physx_lateral_acceleration_mps2 = plant_.lateral_acceleration_mps2;
  observation.physx_rear_axle_x_m = plant_.rear_axle_x_m;
  observation.physx_rear_axle_y_m = plant_.rear_axle_y_m;
  observation.physx_pose_valid = true;
  std::copy(plant_.state.begin() + 3, plant_.state.begin() + 7,
    observation.physx_wheel_omega_radps.begin());
  observation.physx_wheel_rotation_angle_rad = plant_.wheel_rotation_angle_rad;
  observation.physx_wheel_encoder_angle_prediction_rad =
    plant_.wheel_encoder_angle_prediction_rad;
  observation.physx_state = plant_.state;
  observation.physx_longitudinal_slip = plant_.longitudinal_slip;
  observation.physx_lateral_slip = plant_.lateral_slip;
  observation.physx_normal_load_n = plant_.normal_load_n;
  observation.physx_state_valid = true;

  start = std::chrono::steady_clock::now();
  result.estimate = observer_.update(observation);
  result.observer_update_us = microseconds_since(start);

  const auto & estimate = result.estimate;
  start = std::chrono::steady_clock::now();
  plant_.state = runtime_.correct(
    estimate.body_u_mps, estimate.body_v_com_mps,
    estimate.yaw_rate_radps, estimate.yaw_rad,
    estimate.roll_rad, estimate.pitch_rad,
    estimate.unity_state[8],
    estimate.roll_rate_radps, estimate.pitch_rate_radps,
    estimate.wheel_omega_radps);
  result.plant_correction_us = microseconds_since(start);
  plant_.longitudinal_acceleration_mps2 = estimate.longitudinal_acceleration_mps2;
  plant_.lateral_acceleration_mps2 = estimate.lateral_acceleration_mps2;
  return result;
}

}  // namespace f1tenth_localization

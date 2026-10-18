#pragma once

#include <array>
#include <cstddef>

namespace f1tenth_localization
{

struct OdometryObserverConfig
{
  double wheel_radius_m{0.059};
  double encoder_ticks_per_revolution{1920.0};
  double imu_reference_forward_m{0.15532};
  double maximum_packet_dt_s{0.10};
  double lateral_acceleration_residual_gain{1.0};
};

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
  double roll_rad{0.0};
  double pitch_rad{0.0};
  double roll_rate_radps{0.0};
  double pitch_rate_radps{0.0};
  double steering_feedback_rad{0.0};
  double throttle_feedback{0.0};
  bool actuator_feedback_valid{false};
  double steering_command_normalized{0.0};
  double throttle_command_normalized{0.0};
  bool actuator_command_valid{false};
  double physx_forward_velocity_mps{0.0};
  double physx_lateral_velocity_com_mps{0.0};
  double physx_yaw_rate_radps{0.0};
  double physx_longitudinal_acceleration_mps2{0.0};
  double physx_lateral_acceleration_mps2{0.0};
  double physx_rear_axle_x_m{0.0};
  double physx_rear_axle_y_m{0.0};
  std::array<double, 4> physx_wheel_omega_radps{};
  std::array<double, 4> physx_wheel_rotation_angle_rad{};
  std::array<double, 4> physx_wheel_encoder_angle_prediction_rad{};
  std::array<double, 26> physx_state{};
  std::array<double, 4> physx_longitudinal_slip{};
  std::array<double, 4> physx_lateral_slip{};
  std::array<double, 4> physx_normal_load_n{};
  bool physx_state_valid{false};
  bool physx_pose_valid{false};
};

struct OdometryEstimate
{
  double stamp_s{0.0};
  double dt_s{0.0};
  double speed_pred_mps{0.0};
  double speed_mps{0.0};
  double body_u_mps{0.0};
  double body_v_mps{0.0};
  double body_v_com_mps{0.0};
  double x_m{0.0};
  double y_m{0.0};
  double physx_model_x_m{0.0};
  double physx_model_y_m{0.0};
  double yaw_rad{0.0};
  double left_angle_rad{0.0};
  double right_angle_rad{0.0};
  double imu_yaw_rad{0.0};
  double encoder_rear_axle_speed_mps{0.0};
  double ax_mps2{0.0};
  double ay_mps2{0.0};
  double yaw_rate_radps{0.0};
  std::array<double, 4> wheel_omega_radps{};
  std::array<double, 2> encoder_model_rotation_residual_rad{};
  std::array<double, 2> wheel_omega_correction_radps{};
  std::array<double, 2> encoder_wheel_omega_average_radps{};
  std::array<double, 4> longitudinal_slip{};
  std::array<double, 4> lateral_slip{};
  std::array<double, 4> normal_load_n{};
  std::array<double, 26> unity_state{};
  double roll_rad{0.0};
  double pitch_rad{0.0};
  double roll_rate_radps{0.0};
  double pitch_rate_radps{0.0};
  double longitudinal_acceleration_mps2{0.0};
  double lateral_acceleration_mps2{0.0};
  double lateral_acceleration_residual_gain{1.0};
  bool valid{false};
};

class OdometryObserver final
{
public:
  explicit OdometryObserver(OdometryObserverConfig config = {});
  void reset() noexcept;
  OdometryEstimate update(const OdometryObservation & observation);

private:
  OdometryEstimate current_estimate() const noexcept;
  OdometryObserverConfig config_;
  bool initialized_{false};
  double previous_stamp_s_{0.0};
  double previous_yaw_rad_{0.0};
  double previous_yaw_rate_radps_{0.0};
  double previous_encoder_left_angle_rad_{0.0};
  double previous_encoder_right_angle_rad_{0.0};
  std::array<double, 4> previous_physx_wheel_encoder_angle_prediction_rad_{};
  double physx_pose_origin_x_m_{0.0};
  double physx_pose_origin_y_m_{0.0};
  bool physx_pose_origin_initialized_{false};
  double body_u_mps_{0.0};
  double body_v_com_mps_{0.0};
  double body_v_mps_{0.0};
  double x_m_{0.0};
  double y_m_{0.0};
  OdometryEstimate last_{};
};

}  // namespace f1tenth_localization

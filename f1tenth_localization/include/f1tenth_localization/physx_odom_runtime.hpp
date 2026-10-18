#pragma once

#include <array>

#include "unity_source_plant.hpp"
#include "f1tenth_localization/unity_wheel_encoder.hpp"

namespace f1tenth_localization
{

struct PredictedPlantState
{
  std::array<double, 26> state{};
  // Rear-axle point transformed from the actor pose after all 1 ms PhysX
  // scene steps, expressed in ROS map axes (forward, left).
  double rear_axle_x_m{0.0};
  double rear_axle_y_m{0.0};
  std::array<double, 4> wheel_rotation_angle_rad{};
  // Predicted WheelEncoder angle: integrates wheel RPM sampled before each
  // Unity fixed-step solve, unlike PhysX's post-wheel-update pose angle.
  std::array<double, 4> wheel_encoder_angle_prediction_rad{};
  std::array<double, 4> longitudinal_slip{};
  std::array<double, 4> lateral_slip{};
  std::array<double, 4> normal_load_n{};
  std::array<double, 4> suspension_force_n{};
  std::array<double, 4> suspension_jounce_rate_mps{};
  std::array<double, 4> tire_longitudinal_force_n{};
  std::array<double, 4> tire_lateral_force_n{};
  std::array<double, 2> delayed_steering_commands_normalized{};
  std::array<double, 2> delayed_throttle_commands_normalized{};
  double longitudinal_acceleration_mps2{0.0};
  double lateral_acceleration_mps2{0.0};
  double yaw_acceleration_radps2{0.0};
  bool suspension_history_initialized{true};
};

// In-process, source-derived 1 kHz vehicle plant used by production odometry.
// It has no subprocess or PhysX-library dependency. Each sensor packet
// advances 25 Unity 1 ms steps, consuming the oldest queued actuator sample.
class SourceDerivedOdomRuntime
{
public:
  explicit SourceDerivedOdomRuntime(double rear_axle_from_com_m);

  PredictedPlantState advance(
    double steering_command_norm, double throttle_command_norm,
    bool steering_feedback_valid, double steering_feedback_rad);

  void correct(
    double body_u_mps, double body_v_com_mps, double yaw_rate_radps,
    double yaw_rad, double roll_rad, double pitch_rad,
    double body_vertical_velocity_world_mps,
    double roll_rate_radps, double pitch_rate_radps,
    const std::array<double, 4> & wheel_omega_radps);

private:
  f1tenth_mpc::UnitySourcePlantState plant_state_{};
  std::array<double, 2> delayed_steering_commands_norm_{};
  std::array<double, 2> delayed_throttle_commands_norm_{};
  std::array<detail::UnityWheelEncoderPrediction, 4> wheel_encoder_prediction_{};
  double rear_axle_from_com_m_{0.15532};
  double yaw_rad_{0.0};
  double rear_axle_x_m_{0.0};
  double rear_axle_y_m_{0.0};
};

}  // namespace f1tenth_localization

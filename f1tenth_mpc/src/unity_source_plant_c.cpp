#include "unity_source_plant_c.h"

#include "unity_source_plant.hpp"

#include <algorithm>
#include <array>

namespace {

f1tenth_mpc::UnitySourcePlantState from_c(
    const double values[UNITY_SOURCE_PLANT_STATE_SIZE], int history_initialized)
{
  f1tenth_mpc::UnitySourcePlantState state;
  state.u_mps = values[UNITY_SOURCE_U];
  state.v_mps = values[UNITY_SOURCE_V];
  state.yaw_rate_radps = values[UNITY_SOURCE_YAW_RATE];
  std::copy_n(values + UNITY_SOURCE_WHEEL_OMEGA_FL, 4,
              state.wheel_omega_radps.begin());
  state.wheel_rotation_angle_rad = {0.0, 0.0, 0.0, 0.0};
  state.body_height_m = values[UNITY_SOURCE_BODY_HEIGHT];
  state.body_height_rate_mps = values[UNITY_SOURCE_BODY_HEIGHT_RATE];
  state.roll_rad = values[UNITY_SOURCE_ROLL];
  state.roll_rate_radps = values[UNITY_SOURCE_ROLL_RATE];
  state.pitch_rad = values[UNITY_SOURCE_PITCH];
  state.pitch_rate_radps = values[UNITY_SOURCE_PITCH_RATE];
  state.physical_steering_rad = values[UNITY_SOURCE_PHYSICAL_STEERING];
  std::copy_n(values + UNITY_SOURCE_SUSPENSION_JOUNCE_FL, 4,
              state.suspension_jounce_m.begin());
  std::copy_n(values + UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL, 4,
              state.low_forward_speed_time_s.begin());
  std::copy_n(values + UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL, 4,
              state.low_side_speed_time_s.begin());
  state.suspension_history_initialized = history_initialized != 0;
  return state;
}

void to_c(const f1tenth_mpc::UnitySourcePlantOutput &source,
          UnitySourcePlantOutputC *output)
{
  const auto &state = source.state;
  output->state[UNITY_SOURCE_U] = state.u_mps;
  output->state[UNITY_SOURCE_V] = state.v_mps;
  output->state[UNITY_SOURCE_YAW_RATE] = state.yaw_rate_radps;
  std::copy(state.wheel_omega_radps.begin(), state.wheel_omega_radps.end(),
            output->state + UNITY_SOURCE_WHEEL_OMEGA_FL);
  std::copy(state.wheel_rotation_angle_rad.begin(),
            state.wheel_rotation_angle_rad.end(),
            output->wheel_rotation_angle_rad);
  output->state[UNITY_SOURCE_BODY_HEIGHT] = state.body_height_m;
  output->state[UNITY_SOURCE_BODY_HEIGHT_RATE] = state.body_height_rate_mps;
  output->state[UNITY_SOURCE_ROLL] = state.roll_rad;
  output->state[UNITY_SOURCE_ROLL_RATE] = state.roll_rate_radps;
  output->state[UNITY_SOURCE_PITCH] = state.pitch_rad;
  output->state[UNITY_SOURCE_PITCH_RATE] = state.pitch_rate_radps;
  output->state[UNITY_SOURCE_PHYSICAL_STEERING] =
      state.physical_steering_rad;
  std::copy(state.suspension_jounce_m.begin(), state.suspension_jounce_m.end(),
            output->state + UNITY_SOURCE_SUSPENSION_JOUNCE_FL);
  std::copy(state.low_forward_speed_time_s.begin(),
            state.low_forward_speed_time_s.end(),
            output->state + UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL);
  std::copy(state.low_side_speed_time_s.begin(),
            state.low_side_speed_time_s.end(),
            output->state + UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL);
  output->suspension_history_initialized =
      state.suspension_history_initialized ? 1 : 0;
  std::copy(source.longitudinal_slip.begin(), source.longitudinal_slip.end(),
            output->longitudinal_slip);
  std::copy(source.maximum_positive_longitudinal_slip.begin(),
            source.maximum_positive_longitudinal_slip.end(),
            output->maximum_positive_longitudinal_slip);
  std::copy(source.maximum_longitudinal_slip_excess_mps.begin(),
            source.maximum_longitudinal_slip_excess_mps.end(),
            output->maximum_longitudinal_slip_excess_mps);
  std::copy(source.lateral_slip.begin(), source.lateral_slip.end(),
            output->lateral_slip);
  std::copy(source.normal_load_n.begin(), source.normal_load_n.end(),
            output->normal_load_n);
  std::copy(source.suspension_force_n.begin(), source.suspension_force_n.end(),
            output->suspension_force_n);
  std::copy(source.suspension_jounce_rate_mps.begin(),
            source.suspension_jounce_rate_mps.end(),
            output->suspension_jounce_rate_mps);
  std::copy(source.tire_longitudinal_force_n.begin(),
            source.tire_longitudinal_force_n.end(),
            output->tire_longitudinal_force_n);
  std::copy(source.tire_lateral_force_n.begin(),
            source.tire_lateral_force_n.end(), output->tire_lateral_force_n);
  output->longitudinal_acceleration_mps2 =
      source.longitudinal_acceleration_mps2;
  output->lateral_acceleration_mps2 = source.lateral_acceleration_mps2;
  output->yaw_acceleration_radps2 = source.yaw_acceleration_radps2;
}

}  // namespace

extern "C" int unity_source_plant_step_c(
    const UnitySourcePlantInputC *input, double duration_s,
    UnitySourcePlantOutputC *output)
{
  if (!input || !output) return 0;
  const f1tenth_mpc::UnitySourcePlantInput plant_input{
      input->steering_target_rad, input->throttle_normalized};
  auto plant_state = from_c(input->state, input->suspension_history_initialized);
  std::copy_n(input->wheel_rotation_angle_rad, 4,
              plant_state.wheel_rotation_angle_rad.begin());
  f1tenth_mpc::UnitySourcePlantOutput plant_output;
  if (!f1tenth_mpc::unity_source_plant_step(
          plant_state,
          plant_input, duration_s, &plant_output)) {
    return 0;
  }
  to_c(plant_output, output);
  return 1;
}

extern "C" int unity_source_plant_slip_excess_c(
    const double state[UNITY_SOURCE_PLANT_STATE_SIZE],
    double longitudinal_slip[4], double excess_speed_mps[4],
    double excess_derivative_wheel_omega[4])
{
  if (!state || !longitudinal_slip || !excess_speed_mps ||
      !excess_derivative_wheel_omega) return 0;
  const auto source_state = from_c(state, 0);
  std::array<double, 4> slip{}, excess{}, derivative{};
  if (!f1tenth_mpc::unity_source_plant_slip_excess(
          source_state, &slip, &excess, &derivative)) return 0;
  std::copy(slip.begin(), slip.end(), longitudinal_slip);
  std::copy(excess.begin(), excess.end(), excess_speed_mps);
  std::copy(derivative.begin(), derivative.end(),
            excess_derivative_wheel_omega);
  return 1;
}

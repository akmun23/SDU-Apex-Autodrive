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
  state.body_height_m = values[UNITY_SOURCE_BODY_HEIGHT];
  state.body_height_rate_mps = values[UNITY_SOURCE_BODY_HEIGHT_RATE];
  state.roll_rad = values[UNITY_SOURCE_ROLL];
  state.roll_rate_radps = values[UNITY_SOURCE_ROLL_RATE];
  state.pitch_rad = values[UNITY_SOURCE_PITCH];
  state.pitch_rate_radps = values[UNITY_SOURCE_PITCH_RATE];
  state.physical_steering_rad = values[UNITY_SOURCE_PHYSICAL_STEERING];
  std::copy_n(values + UNITY_SOURCE_SUSPENSION_JOUNCE_FL, 4,
              state.suspension_jounce_m.begin());
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
  output->suspension_history_initialized =
      state.suspension_history_initialized ? 1 : 0;
  std::copy(source.longitudinal_slip.begin(), source.longitudinal_slip.end(),
            output->longitudinal_slip);
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
  f1tenth_mpc::UnitySourcePlantOutput plant_output;
  if (!f1tenth_mpc::unity_source_plant_step(
          from_c(input->state, input->suspension_history_initialized),
          plant_input, duration_s, &plant_output)) {
    return 0;
  }
  to_c(plant_output, output);
  return 1;
}

#include "f1tenth_mpc/vehicle_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace f1tenth_mpc {

double clamp(const double value, const double lower, const double upper)
{
  return std::max(lower, std::min(upper, value));
}

double wrap_angle(double angle)
{
  constexpr double pi = 3.14159265358979323846;
  while (angle > pi) angle -= 2.0 * pi;
  while (angle < -pi) angle += 2.0 * pi;
  return angle;
}

VehicleModel::VehicleModel(VehicleModelConfig config)
: config_(std::move(config))
{
  if (!(config_.wheelbase_m > 0.0) || !(config_.max_speed_mps > 0.0) ||
      !(config_.max_steering_rad > 0.0) ||
      !(config_.min_speed_mps >= 0.0) ||
      !(config_.max_acceleration_mps2 > config_.min_acceleration_mps2) ||
      !(config_.cg_to_front_axle_m > 0.0) || !(config_.cg_to_rear_axle_m > 0.0) ||
      !(config_.cg_height_m > 0.0) || !(config_.mass_kg > 0.0) ||
      !(config_.yaw_inertia_kgm2 > 0.0) || !(config_.friction_coefficient > 0.0) ||
      !(config_.front_cornering_stiffness > 0.0) ||
      !(config_.rear_cornering_stiffness > 0.0) || !(config_.gravity_mps2 > 0.0) ||
      !(config_.minimum_slip_velocity_mps > 0.0)) {
    throw std::invalid_argument("vehicle model has non-positive physical parameter");
  }
}

ModelState VehicleModel::step(
    const ModelState &state, const ModelInput &input, const double dt) const
{
  if (!(dt > 0.0) || !std::isfinite(dt)) {
    throw std::invalid_argument("vehicle model dt must be finite and positive");
  }

  ModelState next = state;
  const double h = dt;
  const double delta = clamp(
    input.steering_rad, -config_.max_steering_rad, config_.max_steering_rad);
  const double acceleration = clamp(
    input.acceleration_mps2, config_.min_acceleration_mps2,
    config_.max_acceleration_mps2);
  const double vx = state.u;
  const double vy = state.v;
  const double omega = state.r;
  const double vx_safe = std::max(vx, config_.minimum_slip_velocity_mps);
  const double front_slip = delta - std::atan(
    (vy + config_.cg_to_front_axle_m * omega) / vx_safe);
  const double rear_slip = -std::atan(
    (vy - config_.cg_to_rear_axle_m * omega) / vx_safe);
  const double longitudinal_force = config_.mass_kg * acceleration;
  const double wheelbase = config_.wheelbase_m;
  const double front_load = (
    config_.mass_kg * config_.gravity_mps2 * config_.cg_to_rear_axle_m -
    longitudinal_force * config_.cg_height_m) / wheelbase;
  const double rear_load = (
    config_.mass_kg * config_.gravity_mps2 * config_.cg_to_front_axle_m +
    longitudinal_force * config_.cg_height_m) / wheelbase;
  const double static_front_load = config_.mass_kg * config_.gravity_mps2 *
    config_.cg_to_rear_axle_m / wheelbase;
  const double static_rear_load = config_.mass_kg * config_.gravity_mps2 *
    config_.cg_to_front_axle_m / wheelbase;
  // BachelorProject stores C_alpha as a force scale and normalizes it by
  // the static friction-load scale before multiplying the current load.
  // Keep that normalization explicit; using mu*C_alpha*Fz would not be the
  // same model and materially changes yaw/lateral response.
  const double front_normalized_stiffness = config_.front_cornering_stiffness /
    (config_.friction_coefficient * static_front_load);
  const double rear_normalized_stiffness = config_.rear_cornering_stiffness /
    (config_.friction_coefficient * static_rear_load);
  const double front_force = config_.friction_coefficient *
    front_normalized_stiffness * front_slip * front_load;
  const double rear_force = config_.friction_coefficient *
    rear_normalized_stiffness * rear_slip * rear_load;
  const double cos_delta = std::cos(delta);
  const double sin_delta = std::sin(delta);
  const double dvx = (longitudinal_force - front_force * sin_delta) /
    config_.mass_kg + vy * omega;
  const double dvy = (front_force * cos_delta + rear_force) /
    config_.mass_kg - vx * omega;
  const double domega = (
    config_.cg_to_front_axle_m * front_force * cos_delta -
    config_.cg_to_rear_axle_m * rear_force) / config_.yaw_inertia_kgm2;

  // Match the BachelorProject model's analytical SE(2) pose integration and
  // forward-Euler body-dynamics update. This class is the offline M0 replay
  // implementation; it is not a runtime ground-truth path.
  const double yaw_end = state.yaw + h * omega;
  if (std::abs(omega) < 1e-6) {
    next.x = state.x + h * (vx * std::cos(state.yaw) - vy * std::sin(state.yaw));
    next.y = state.y + h * (vx * std::sin(state.yaw) + vy * std::cos(state.yaw));
  } else {
    const double inv_omega = 1.0 / omega;
    next.x = state.x + (
      vx * (std::sin(yaw_end) - std::sin(state.yaw)) +
      vy * (std::cos(yaw_end) - std::cos(state.yaw))) * inv_omega;
    next.y = state.y + (
      vx * (std::cos(state.yaw) - std::cos(yaw_end)) +
      vy * (std::sin(yaw_end) - std::sin(state.yaw))) * inv_omega;
  }
  next.yaw = wrap_angle(yaw_end);
  next.u = clamp(vx + h * dvx, config_.min_speed_mps, config_.max_speed_mps);
  next.v = vy + h * dvy;
  next.r = omega + h * domega;
  next.steering = delta;
  next.steering_valid = true;
  return next;
}

std::vector<ModelState> VehicleModel::rollout(
    const ModelState &initial,
    const std::vector<ModelInput> &inputs,
    const std::vector<double> &dt) const
{
  if (inputs.size() != dt.size()) {
    throw std::invalid_argument("vehicle model input and dt lengths differ");
  }
  std::vector<ModelState> states;
  states.reserve(inputs.size() + 1);
  states.push_back(initial);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    states.push_back(step(states.back(), inputs[i], dt[i]));
  }
  return states;
}

}  // namespace f1tenth_mpc

#include "f1tenth_mpc/bachelor_mpc_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

extern "C" {
#include "mpc.h"
}

namespace f1tenth_mpc {
namespace {

constexpr double kPi = 3.14159265358979323846;

const char *status_text(const MpcSolverStatus_t status)
{
  switch (status) {
    case MPC_STATUS_SUCCESS: return "success";
    case MPC_STATUS_MAXIMUM_ITERATIONS_REACHED: return "maximum_iterations_reached";
    case MPC_STATUS_INFEASIBLE: return "infeasible";
    case MPC_STATUS_ERROR: return "error";
    default: return "unknown";
  }
}

double wrap_angle_local(double angle)
{
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

bool finite_control(const ControlInput_t &control)
{
  return std::isfinite(control.steer_ang) && std::isfinite(control.long_acc);
}

}  // namespace

struct BachelorMpcController::Impl {
  MpcConfiguration_t native_config{};
  TrajectoryReferencePoint_t reference[PREDICTION_HORIZON]{};
  ControlInput_t previous_control{};
  bool previous_control_valid{false};
  bool initialized{false};
};

BachelorMpcController::BachelorMpcController(
  const TrackModel &track, BachelorMpcConfig config)
: track_(track), config_(config), impl_(new Impl{})
{
  configure(config_);
}

BachelorMpcController::~BachelorMpcController()
{
  delete impl_;
}

void BachelorMpcController::configure(const BachelorMpcConfig &config)
{
  impl_->native_config = get_default_configuration();
  impl_->native_config.time_step = static_cast<float>(config.prediction_dt_s);
  impl_->native_config.weight_lateral_error = static_cast<float>(config.weight_lateral_error);
  impl_->native_config.weight_heading_error = static_cast<float>(config.weight_heading_error);
  impl_->native_config.weight_velocity = static_cast<float>(config.weight_velocity);
  impl_->native_config.weight_lateral_velocity = static_cast<float>(config.weight_lateral_velocity);
  impl_->native_config.weight_yaw_rate = static_cast<float>(config.weight_yaw_rate);
  impl_->native_config.weight_steering_effort = static_cast<float>(config.weight_steering_effort);
  impl_->native_config.weight_acceleration_effort = static_cast<float>(config.weight_acceleration_effort);
  impl_->native_config.weight_steering_rate = static_cast<float>(config.weight_steering_rate);
  impl_->native_config.weight_acceleration_rate = static_cast<float>(config.weight_acceleration_rate);
  impl_->native_config.weight_effective_steering = static_cast<float>(config.weight_effective_steering);
  impl_->native_config.cross_call_rate_scale = static_cast<float>(config.cross_call_rate_scale);
  impl_->native_config.wall_margin = static_cast<float>(config.wall_margin);
  impl_->native_config.max_solver_iterations = static_cast<uint16_t>(std::clamp<std::size_t>(
    config.max_solver_iterations, 1U, std::numeric_limits<uint16_t>::max()));
  impl_->native_config.solver_convergence_tolerance =
    static_cast<float>(config.solver_convergence_tolerance);

  if (!(impl_->native_config.time_step > 0.0F) ||
      !std::isfinite(impl_->native_config.time_step)) {
    impl_->native_config.time_step = TIME_STEP_SECONDS;
  }
  if (!(impl_->native_config.wall_margin >= 0.0F) ||
      !std::isfinite(impl_->native_config.wall_margin)) {
    impl_->native_config.wall_margin = WALL_MARGIN;
  }
  mpc_initialize_with_configuration(&impl_->native_config);
  impl_->previous_control = {};
  impl_->previous_control_valid = false;
  impl_->initialized = true;
}

void BachelorMpcController::build_reference(const TrackModel &, double s, double dt)
{
  // This is the BachelorProject MPC reference construction: each stage is
  // advanced by the previous stage's raceline speed, then interpolated by s.
  double step_velocity = track_.sample(s).speed;
  if (!(step_velocity > 0.0) || !std::isfinite(step_velocity)) {
    step_velocity = MIN_TRAJECTORY_SPEED_MPS;
  }

  double query_s = s;
  for (std::size_t stage = 0; stage < PREDICTION_HORIZON; ++stage) {
    query_s += step_velocity * dt;
    const TrackPoint point = track_.sample(query_s);
    step_velocity = std::isfinite(point.speed) ? std::max(0.0, point.speed) : 0.0;

    auto &reference = impl_->reference[stage];
    reference.reference_lateral_error = 0.0F;
    reference.reference_heading_error = 0.0F;
    reference.reference_velocity = static_cast<float>(step_velocity);
    reference.reference_lateral_velocity = 0.0F;
    reference.reference_yaw_rate = static_cast<float>(point.curvature * step_velocity);
    reference.path_curvature = static_cast<float>(point.curvature);
    reference.left_wall_bound = static_cast<float>(point.left_bound);
    reference.right_wall_bound = static_cast<float>(point.right_bound);
  }
}

BachelorMpcResult BachelorMpcController::solve(const ModelState &state)
{
  BachelorMpcResult output;
  output.horizon_steps = PREDICTION_HORIZON;

  if (!impl_->initialized || track_.empty()) {
    output.status_text = "not_initialized_or_empty_track";
    return output;
  }
  if (!std::isfinite(state.x) || !std::isfinite(state.y) ||
      !std::isfinite(state.yaw) || !std::isfinite(state.u) ||
      !std::isfinite(state.v) || !std::isfinite(state.r)) {
    output.status_text = "non_finite_state";
    return output;
  }

  const TrackProjection projection = track_.project(state.x, state.y, state.yaw);
  if (!projection.valid || !std::isfinite(projection.s)) {
    output.status_text = "track_projection_failed";
    return output;
  }

  const TrackPoint current_point = track_.sample(projection.s);
  const double dt = static_cast<double>(impl_->native_config.time_step);
  build_reference(track_, projection.s, dt);

  FrenetState_t frenet{};
  frenet.flat_error = static_cast<float>(projection.lateral_error);
  frenet.fhead_error = static_cast<float>(
    wrap_angle_local(state.yaw - current_point.yaw));
  frenet.flong_vel = static_cast<float>(std::max(0.0, state.u));
  frenet.flat_vel = static_cast<float>(state.v);
  frenet.fyaw_rate = static_cast<float>(state.r);

  // The core API requires the command that acted during the preceding
  // interval. A valid command echo is preferred; otherwise retain the last
  // command issued by this wrapper. No simulator truth or GT command is used.
  ControlInput_t previous = impl_->previous_control;
  if (state.steering_valid && std::isfinite(state.steering)) {
    previous.steer_ang = static_cast<float>(state.steering);
  }
  if (!impl_->previous_control_valid) {
    previous.long_acc = 0.0F;
  }
  mpc_set_previous_command(&previous);

  MpcSolverResult_t native_result{};
  const MpcSolverStatus_t status = mpc_compute_optimal_control(
    &frenet, impl_->reference, &native_result);
  output.status = static_cast<int>(status);
  output.status_text = status_text(status);
  output.solver_success = status == MPC_STATUS_SUCCESS;
  output.solver_iterations = native_result.iterations_used;
  output.primal_residual = native_result.final_cost;
  output.dual_residual = native_result.dual_residual;
  output.steering_rad = native_result.optimal_control.steer_ang;
  output.acceleration_mps2 = native_result.optimal_control.long_acc;
  output.current_lateral_error_m = projection.lateral_error;
  output.current_heading_error_rad = frenet.fhead_error;
  const bool accepted = status == MPC_STATUS_SUCCESS ||
    (status == MPC_STATUS_MAXIMUM_ITERATIONS_REACHED && config_.accept_max_iterations);
  output.valid = accepted && finite_control(native_result.optimal_control);

  float planned_states[PREDICTION_HORIZON + 1][RICCATI_MAX_NX]{};
  if (mpc_debug_copy_last_plan(planned_states, nullptr)) {
    output.predicted_lateral_error_m = planned_states[PREDICTION_HORIZON][IDX_EY];
    double predicted_distance = 0.0;
    for (std::size_t stage = 1; stage <= PREDICTION_HORIZON; ++stage) {
      const double velocity = std::max(0.0, static_cast<double>(planned_states[stage][2]));
      predicted_distance += velocity * dt;
    }
    output.predicted_progress_m = predicted_distance;
  }

  if (output.valid) {
    impl_->previous_control = native_result.optimal_control;
    impl_->previous_control_valid = true;
  }
  return output;
}

void BachelorMpcController::reset()
{
  mpc_reset();
  impl_->previous_control = {};
  impl_->previous_control_valid = false;
}

}  // namespace f1tenth_mpc

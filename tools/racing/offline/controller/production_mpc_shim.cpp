// Offline C ABI for the production C MPC core. The solver, nonlinear model,
// refinement and acceptance checks are the same sources called by the ROS node.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

extern "C" {
#include "mpc_rti.h"
#include "vehicle_model.h"
}

#include "yaw_response_surface_io.hpp"

namespace
{

constexpr std::size_t kFailureDiagnosticCount = 29;
constexpr std::size_t kStepOutputCount = 16 + 2 * kFailureDiagnosticCount;

void copy_failure_diagnostics(const MpcRtiRolloutFailure_t & failure,
                              double * output)
{
  const auto & state = failure.state;
  const auto & plant = state.plant;
  const auto & reference = failure.reference;
  const double values[kFailureDiagnosticCount] = {
    static_cast<double>(failure.valid),
    static_cast<double>(failure.stage),
    failure.progress,
    plant.e_y, plant.e_psi, plant.u, plant.v, plant.r,
    plant.target_speed, plant.steering_command,
    plant.delayed_steering_command_1,
    plant.delayed_steering_command_2,
    plant.actual_steering_angle,
    state.previous_steering_rate,
    state.previous_target_speed_rate,
    reference.e_y, reference.e_psi, reference.u, reference.v, reference.r,
    reference.steering_command, reference.target_speed,
    reference.target_speed_rate, reference.path_curvature,
    reference.left_bound, reference.right_bound,
    failure.margin_m, failure.lower_bound_m, failure.upper_bound_m,
  };
  std::copy(values, values + kFailureDiagnosticCount, output);
}

template<typename T>
T read_param(const YAML::Node & params, const char * name, T fallback)
{
  const auto value = params[name];
  return value ? value.as<T>() : fallback;
}

float fparam(const YAML::Node & params, const char * name, float fallback)
{
  return static_cast<float>(read_param<double>(params, name, fallback));
}

int iparam(const YAML::Node & params, const char * name, int fallback)
{
  return read_param<int>(params, name, fallback);
}

MpcRtiCycleConfiguration_t load_config(const std::string & path,
                                       double & maximum_speed)
{
  const auto root = YAML::LoadFile(path);
  const auto params = root["/**"]["ros__parameters"];
  if (!params) {
    throw std::runtime_error("/**.ros__parameters not found in MPC YAML");
  }

  maximum_speed = read_param<double>(params, "max_speed_mps", 16.0);
  const auto defaults = vehicle_model_default_yaw_rate_parameters();
  MpcYawRateModelParameters_t yaw{};
  yaw.response_time_constant_s = fparam(params,
      "yaw_rate_response_time_constant_s", defaults.response_time_constant_s);
  yaw.steering_gain_per_m = fparam(params,
      "yaw_rate_steering_gain_per_m", defaults.steering_gain_per_m);
  yaw.steering_gain_reduction_per_rad = fparam(params,
      "yaw_rate_steering_gain_reduction_per_rad",
      defaults.steering_gain_reduction_per_rad);
  yaw.steering_gain_start_rad = fparam(params,
      "yaw_rate_steering_gain_start_rad", defaults.steering_gain_start_rad);
  yaw.steering_gain_end_rad = fparam(params,
      "yaw_rate_steering_gain_end_rad", defaults.steering_gain_end_rad);
  yaw.curvature_gain_reduction_per_m = fparam(params,
      "yaw_rate_curvature_gain_reduction_per_m",
      defaults.curvature_gain_reduction_per_m);
  yaw.curvature_gain_start_per_m = fparam(params,
      "yaw_rate_curvature_gain_start_per_m", defaults.curvature_gain_start_per_m);
  yaw.curvature_gain_end_per_m = fparam(params,
      "yaw_rate_curvature_gain_end_per_m", defaults.curvature_gain_end_per_m);
  yaw.low_speed_response_time_constant_s = fparam(params,
      "yaw_rate_low_speed_response_time_constant_s",
      defaults.low_speed_response_time_constant_s);
  yaw.low_speed_transition_speed_mps = fparam(params,
      "yaw_rate_low_speed_transition_speed_mps",
      defaults.low_speed_transition_speed_mps);
  if (!vehicle_model_set_yaw_rate_parameters(&yaw)) {
    throw std::runtime_error("invalid production yaw-rate model parameters");
  }
  const bool yaw_surface_enabled = read_param<bool>(
      params, "yaw_rate_response_surface_enabled", false);
  if (yaw_surface_enabled) {
    const float blend_start = fparam(params,
        "yaw_rate_response_surface_blend_q_start", 0.60f);
    const float blend_end = fparam(params,
        "yaw_rate_response_surface_blend_q_end", 0.85f);
    const float speed_blend_margin = fparam(params,
        "yaw_rate_response_surface_speed_blend_margin_mps", 0.50f);
    const float low_speed_blend_margin = fparam(params,
        "yaw_rate_response_surface_low_speed_blend_margin_mps",
        speed_blend_margin);
    const float low_speed_support_fadeout = fparam(params,
        "yaw_rate_response_surface_low_speed_support_fadeout_mps", 0.0f);
    const float high_speed_support_fadein = fparam(params,
        "yaw_rate_response_surface_high_speed_support_fadein_mps", 0.0f);
    const auto configured_surface_path = read_param<std::string>(
        params, "yaw_rate_response_surface_file", "");
    auto surface_path = std::filesystem::path(path).parent_path() /
        "yaw_response_surface.csv";
    if (!configured_surface_path.empty()) {
      const std::filesystem::path configured(configured_surface_path);
      surface_path = configured.is_absolute()
          ? configured : std::filesystem::path(path).parent_path() / configured;
    }
    const auto surface = f1tenth_mpc::load_yaw_response_surface_csv(
        surface_path.string(), blend_start, blend_end, speed_blend_margin,
        low_speed_blend_margin, low_speed_support_fadeout,
        high_speed_support_fadein);
    if (!vehicle_model_set_yaw_rate_response_surface(&surface)) {
      throw std::runtime_error("invalid production yaw-rate response surface");
    }
  } else {
    const MpcYawRateResponseSurface_t disabled_surface{};
    (void)vehicle_model_set_yaw_rate_response_surface(&disabled_surface);
  }

  MpcRtiCycleConfiguration_t config{};
  auto & model = config.model;
#define WEIGHT(field, fallback) model.field = fparam(params, #field, fallback)
  WEIGHT(weight_e_y, 125.0f);
  WEIGHT(weight_e_psi, 2.0f);
  WEIGHT(weight_u, 200.0f);
  WEIGHT(weight_u_overspeed, 200.0f);
  WEIGHT(weight_target_speed_state, 20.0f);
  WEIGHT(weight_v, 5.0f);
  WEIGHT(weight_r, 1.5f);
  WEIGHT(weight_steering_command, 3.0f);
  WEIGHT(weight_steering_rate, 5.0f);
  WEIGHT(weight_target_speed_rate, 0.5f);
  WEIGHT(weight_steering_rate_change, 10.0f);
  WEIGHT(weight_target_speed_rate_change, 5.0f);
  WEIGHT(terminal_multiplier, 3.0f);
  model.max_speed_mps = static_cast<float>(maximum_speed);
  model.active_speed_ceiling_mps = static_cast<float>(maximum_speed);
  WEIGHT(max_steering_rad, SOURCE_MAX_STEERING_RAD);
  WEIGHT(max_steering_rate_radps, SOURCE_STEERING_RATE_RADPS);
  WEIGHT(max_target_speed_rate_increase_mps2,
         MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2);
  WEIGHT(max_target_speed_rate_reduction_mps2,
         MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2);
  WEIGHT(corridor_margin_m, MPC_REQUIRED_WALL_CLEARANCE_M);
  model.first_prediction_corridor_margin_m = fparam(params,
      "first_prediction_corridor_margin_m", model.corridor_margin_m);
  WEIGHT(corridor_preview_halfwidth_m, 0.10f);
  WEIGHT(nonlinear_corridor_tolerance_m, 0.001f);
  model.use_fd_jacobian_oracle = 0;
  const auto seed_policy = read_param<std::string>(
      params, "recovery_seed_policy", "nominal");
  if (seed_policy == "nominal") {
    model.recovery_seed_policy = MPC_RTI_RECOVERY_SEED_NOMINAL;
  } else if (seed_policy == "heading_feedback") {
    model.recovery_seed_policy = MPC_RTI_RECOVERY_SEED_HEADING_FEEDBACK;
  } else if (seed_policy == "brake_heading_feedback") {
    model.recovery_seed_policy = MPC_RTI_RECOVERY_SEED_BRAKE_HEADING_FEEDBACK;
  } else {
    throw std::runtime_error("invalid recovery_seed_policy in MPC YAML");
  }
  model.recovery_steering_k_e_y = fparam(params, "recovery_steering_k_e_y", 0.0f);
  model.recovery_steering_k_e_psi = fparam(params, "recovery_steering_k_e_psi", 0.0f);
  model.recovery_steering_k_r = fparam(params, "recovery_steering_k_r", 0.0f);
#undef WEIGHT

  auto & solver = config.solver;
  solver.max_iterations = static_cast<uint16_t>(std::clamp(
      iparam(params, "max_solver_iterations", 100), 1, 100));
  solver.rho = fparam(params, "admm_rho", 7.0f);
  solver.rho_u = fparam(params, "admm_rho_input", 7.0f);
  solver.tolerance = fparam(params, "solver_tolerance", 0.01f);
  solver.adaptive_rho = read_param<bool>(params, "adaptive_rho", true) ? 1 : 0;
  solver.shared_rho = 0;
  solver.over_relaxation = fparam(params, "admm_over_relaxation", 1.6f);
  solver.use_prefactorization = read_param<bool>(
      params, "use_riccati_prefactorization", true) ? 1 : 0;

  const auto refinement = read_param<std::string>(
      params, "rti_refinement_mode", "adaptive");
  if (refinement == "r1") config.refinement_mode = MPC_RTI_REFINEMENT_R1;
  else if (refinement == "r2") config.refinement_mode = MPC_RTI_REFINEMENT_R2;
  else if (refinement == "adaptive" || refinement == "ra")
    config.refinement_mode = MPC_RTI_REFINEMENT_ADAPTIVE;
  else throw std::runtime_error("invalid rti_refinement_mode in MPC YAML");
  config.rti2_progress_error_trigger_m = fparam(
      params, "rti2_progress_error_trigger_m", 0.10f);
  config.rti2_curvature_error_trigger_per_m = fparam(
      params, "rti2_curvature_error_trigger_per_m", 0.02f);
  config.rti2_bound_error_trigger_m = fparam(
      params, "rti2_bound_error_trigger_m", 0.05f);
  config.rti2_min_corridor_slack_trigger_m = fparam(
      params, "rti2_min_corridor_slack_trigger_m", 0.25f);
  config.rti2_steering_rate_correction_trigger_radps = fparam(
      params, "rti2_steering_rate_correction_trigger_radps", 0.50f);
  config.rti2_target_speed_rate_correction_trigger_mps2 = fparam(
      params, "rti2_target_speed_rate_correction_trigger_mps2", 1.0f);
  config.rti2_residual_imbalance_trigger = fparam(
      params, "rti2_residual_imbalance_trigger", 8.0f);
  config.rti2_lateral_load_trigger_mps2 = fparam(
      params, "rti2_lateral_load_trigger_mps2", 3.0f);
  config.rti2_nonsmooth_columns_trigger = iparam(
      params, "rti2_nonsmooth_columns_trigger", 50);
  config.rti2_residual_recovery_limit = fparam(
      params, "rti2_residual_recovery_limit", 0.25f);
  config.degraded_residual_limit = fparam(
      params, "solver_degraded_tolerance", 0.01f);
  config.maximum_regularization = fparam(
      params, "solver_max_regularization", 0.01f);
  config.max_consecutive_degraded_solves = iparam(
      params, "max_consecutive_degraded_solves", 3);
  return config;
}

std::vector<MpcTrajectorySample_t> load_trajectory(const std::string & path)
{
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open MPC trajectory: " + path);
  std::vector<MpcTrajectorySample_t> points;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    std::vector<double> values;
    std::size_t begin = 0;
    bool valid = true;
    for (;;) {
      const auto comma = line.find(',', begin);
      try {
        values.push_back(std::stod(line.substr(begin, comma - begin)));
      } catch (...) {
        valid = false;
        break;
      }
      if (comma == std::string::npos) break;
      begin = comma + 1;
    }
    if (!valid || values.size() < 6) continue;
    MpcTrajectorySample_t point{};
    point.s = values[0]; point.x = values[1]; point.y = values[2];
    point.heading = values[3]; point.curvature = values[4];
    point.speed = values[5]; point.acceleration = values.size() >= 7 ? values[6] : 0.0;
    point.left_bound = values.size() >= 9 ? values[7] : 1.0;
    point.right_bound = values.size() >= 9 ? values[8] : 1.0;
    if (std::isfinite(point.s) && std::isfinite(point.x) &&
        std::isfinite(point.y) && std::isfinite(point.heading) &&
        std::isfinite(point.curvature) && std::isfinite(point.speed) &&
        std::isfinite(point.acceleration) && point.left_bound > 0.0 &&
        point.right_bound > 0.0) {
      points.push_back(point);
    }
  }
  return points;
}

class ProductionMpc
{
public:
  ProductionMpc(const std::string & yaml, const std::string & trajectory)
  : config_(load_config(yaml, maximum_speed_)),
    trajectory_(load_trajectory(trajectory))
  {
    if (trajectory_.empty()) throw std::runtime_error("empty MPC trajectory");
    point_count_ = trajectory_.size();
    if (!mpc_trajectory_prepare(trajectory_.data(), &point_count_, &lap_length_)) {
      throw std::runtime_error("production MPC trajectory preparation failed");
    }
    trajectory_.resize(point_count_);
    mpc_rti_memory_reset(&memory_);
  }

  void reset() { mpc_rti_memory_reset(&memory_); }

  bool project(double x, double y, double yaw, std::size_t previous_segment,
               std::size_t local_search_radius, double * out,
               std::size_t out_size) const
  {
    if (!out || out_size < 5) return false;
    MpcPathProjection_t projection{};
    if (!mpc_trajectory_project(
        trajectory_.data(), trajectory_.size(), lap_length_, x, y, yaw,
        previous_segment, local_search_radius, &projection)) return false;
    out[0] = projection.s;
    out[1] = projection.lateral_error;
    out[2] = projection.heading_error;
    out[3] = projection.distance;
    out[4] = static_cast<double>(projection.segment);
    return true;
  }

  bool step(const double * state, double progress, double speed_ceiling,
            double * out, std::size_t out_size)
  {
    if (!state || !out || out_size < kStepOutputCount || !std::isfinite(progress) ||
        !std::isfinite(speed_ceiling) || speed_ceiling <= 0.0 ||
        speed_ceiling > maximum_speed_) return false;
    MpcRtiState_t current{};
    current.plant.e_y = static_cast<float>(state[0]);
    current.plant.e_psi = static_cast<float>(state[1]);
    current.plant.u = static_cast<float>(state[2]);
    current.plant.v = static_cast<float>(state[3]);
    current.plant.r = static_cast<float>(state[4]);
    current.plant.target_speed = static_cast<float>(state[5]);
    current.plant.steering_command = static_cast<float>(state[6]);
    current.plant.delayed_steering_command_1 = static_cast<float>(state[7]);
    current.plant.delayed_steering_command_2 = static_cast<float>(state[8]);
    current.plant.actual_steering_angle = static_cast<float>(state[9]);
    current.previous_steering_rate = static_cast<float>(state[10]);
    current.previous_target_speed_rate = static_cast<float>(state[11]);
    config_.model.active_speed_ceiling_mps = static_cast<float>(speed_ceiling);
    MpcRtiCycleResult_t result{};
    const auto status = mpc_rti_solve_cycle(
        &current, progress, trajectory_.data(), trajectory_.size(), lap_length_,
        TIME_STEP_SECONDS, PREDICTION_HORIZON, &config_, &memory_, &result);
    const double values[16] = {
      static_cast<double>(status), result.published_steering_command,
      result.published_target_speed, result.first_control.steering_rate,
      result.first_control.target_speed_rate,
      static_cast<double>(result.solver_iterations), result.primal_residual,
      result.dual_residual, result.maximum_regularization,
      static_cast<double>(result.nonlinear_failure_stage),
      static_cast<double>(result.nonlinear_failure_reason),
      static_cast<double>(result.best_effort_action_published),
      static_cast<double>(result.residual_candidate_published),
      static_cast<double>(result.rejection_speed_guard_applied),
      static_cast<double>(result.rti_iterations_used),
      static_cast<double>(result.rti2_triggered),
    };
    std::copy(values, values + 16, out);
    copy_failure_diagnostics(result.r1_nonlinear_failure, out + 16);
    copy_failure_diagnostics(result.r2_nonlinear_failure,
                             out + 16 + kFailureDiagnosticCount);
    return true;
  }

  double lap_length() const { return lap_length_; }

private:
  MpcRtiCycleConfiguration_t config_{};
  double maximum_speed_{16.0};
  std::vector<MpcTrajectorySample_t> trajectory_;
  std::size_t point_count_{0};
  double lap_length_{0.0};
  MpcRtiMemory_t memory_{};
};

}  // namespace

extern "C"
{

void * offline_mpc_create(const char * yaml_path, const char * trajectory_path,
                          char * error, std::size_t error_capacity)
{
  try {
    return new ProductionMpc(yaml_path, trajectory_path);
  } catch (const std::exception & exception) {
    if (error && error_capacity > 0) {
      const auto length = std::min(error_capacity - 1,
          std::char_traits<char>::length(exception.what()));
      std::copy(exception.what(), exception.what() + length, error);
      error[length] = '\0';
    }
    return nullptr;
  }
}

void offline_mpc_destroy(void * handle)
{
  delete static_cast<ProductionMpc *>(handle);
}

void offline_mpc_reset(void * handle)
{
  static_cast<ProductionMpc *>(handle)->reset();
}

int offline_mpc_step(void * handle, const double * state, double progress,
                     double speed_ceiling, double * output,
                     std::size_t output_count)
{
  return static_cast<ProductionMpc *>(handle)->step(
      state, progress, speed_ceiling, output, output_count) ? 1 : 0;
}

int offline_mpc_project(void * handle, double x, double y, double yaw,
                        std::size_t previous_segment,
                        std::size_t local_search_radius, double * output,
                        std::size_t output_count)
{
  return static_cast<ProductionMpc *>(handle)->project(
      x, y, yaw, previous_segment, local_search_radius,
      output, output_count) ? 1 : 0;
}

double offline_mpc_lap_length(void * handle)
{
  return static_cast<ProductionMpc *>(handle)->lap_length();
}

}

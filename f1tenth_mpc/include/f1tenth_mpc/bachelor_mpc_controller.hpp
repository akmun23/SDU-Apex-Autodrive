#pragma once

#include "f1tenth_mpc/track_model.hpp"
#include "f1tenth_mpc/vehicle_model.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace f1tenth_mpc {

// Configuration exposed by the ROS wrapper. The vehicle model, augmented
// state, 20-stage horizon, and solver implementation remain the exact
// BachelorProject MPC defaults; only the public runtime tuning fields are
// configurable here.
struct BachelorMpcConfig {
  double prediction_dt_s{0.03};
  double weight_lateral_error{1500.0};
  double weight_heading_error{50.0};
  double weight_velocity{200.0};
  double weight_lateral_velocity{5.0};
  double weight_yaw_rate{1.5};
  double weight_steering_effort{2.0};
  double weight_acceleration_effort{0.5};
  double weight_steering_rate{5.0};
  double weight_acceleration_rate{5.0};
  double weight_effective_steering{1.0};
  double cross_call_rate_scale{1.0 / 6.0};
  double wall_margin{0.0};
  std::size_t max_solver_iterations{50};
  double solver_convergence_tolerance{0.01};
  bool accept_max_iterations{true};
};

struct BachelorMpcResult {
  bool valid{false};
  bool solver_success{false};
  int status{3};
  std::string status_text{"not_initialized"};
  double steering_rad{0.0};
  double acceleration_mps2{0.0};
  double current_lateral_error_m{0.0};
  double current_heading_error_rad{0.0};
  double predicted_lateral_error_m{0.0};
  double predicted_progress_m{0.0};
  double primal_residual{0.0};
  double dual_residual{0.0};
  std::size_t horizon_steps{0};
  std::size_t solver_iterations{0};
};

class BachelorMpcController {
public:
  BachelorMpcController(const TrackModel &track, BachelorMpcConfig config = {});
  ~BachelorMpcController();

  BachelorMpcController(const BachelorMpcController &) = delete;
  BachelorMpcController &operator=(const BachelorMpcController &) = delete;

  BachelorMpcResult solve(const ModelState &state);
  void reset();

private:
  void configure(const BachelorMpcConfig &config);
  void build_reference(const TrackModel &track, double s, double dt);

  TrackModel track_;
  BachelorMpcConfig config_;
  struct Impl;
  Impl *impl_{nullptr};
};

}  // namespace f1tenth_mpc

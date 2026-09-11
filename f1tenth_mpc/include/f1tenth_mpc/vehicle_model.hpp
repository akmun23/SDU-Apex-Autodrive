#pragma once

#include <string>
#include <vector>

namespace f1tenth_mpc {

struct ModelState {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double u{0.0};
  double v{0.0};
  double r{0.0};
  double steering{0.0};
  bool steering_valid{false};
};

struct ModelInput {
  // Direct physical inputs used by the BachelorProject plant model.
  // These are not target-speed or target-steering commands.
  double steering_rad{0.0};
  double acceleration_mps2{0.0};
};

struct VehicleModelConfig {
  // This is the deterministic BachelorProject dynamic-bicycle baseline.  It
  // is a model baseline, not an identified simulator model; identification
  // must replace these values only after source-time data passes its gates.
  std::string version{"bachelor_mpc_dynamic_bicycle_baseline_v1"};
  double wheelbase_m{0.326};
  double max_speed_mps{20.0};
  double max_steering_rad{0.39};
  double min_speed_mps{0.5};
  double max_acceleration_mps2{0.72 * 9.82};
  double min_acceleration_mps2{-0.72 * 9.82};
  double cg_to_front_axle_m{0.166};
  double cg_to_rear_axle_m{0.16};
  double cg_height_m{0.0703};
  double mass_kg{3.314};
  double yaw_inertia_kgm2{0.035};
  double friction_coefficient{0.72};
  double front_cornering_stiffness{51.40};
  double rear_cornering_stiffness{43.10};
  double gravity_mps2{9.82};
  double minimum_slip_velocity_mps{0.5};
};

class VehicleModel {
public:
  explicit VehicleModel(VehicleModelConfig config = {});

  const VehicleModelConfig &config() const { return config_; }
  ModelState step(const ModelState &state, const ModelInput &input, double dt) const;
  std::vector<ModelState> rollout(
      const ModelState &initial,
      const std::vector<ModelInput> &inputs,
      const std::vector<double> &dt) const;

private:
  VehicleModelConfig config_;
};

double wrap_angle(double angle);
double clamp(double value, double lower, double upper);

}  // namespace f1tenth_mpc

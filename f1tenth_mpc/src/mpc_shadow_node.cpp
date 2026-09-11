#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "f1tenth_mpc/bachelor_mpc_controller.hpp"
#include "mpc_types.h"
#include "rclcpp/rclcpp.hpp"
#include "sdu_apex_msgs/msg/mpc_diagnostics.hpp"
#include "sdu_apex_msgs/msg/vehicle_control_state.hpp"

namespace f1tenth_mpc {

class MpcShadowNode final : public rclcpp::Node {
public:
  MpcShadowNode()
  : Node("mpc_shadow_node")
  {
    trajectory_file_ = declare_parameter("trajectory_file", std::string{});
    state_topic_ = declare_parameter("state_topic", std::string("/mpc/control_state"));
    diagnostics_topic_ = declare_parameter(
      "diagnostics_topic", std::string("/mpc/diagnostics"));
    shadow_command_topic_ = declare_parameter(
      "shadow_command_topic", std::string("/mpc/shadow_command"));
    model_version_ = declare_parameter(
      "model_version", std::string("bachelor_mpc_riccati_augmented9_v1"));

    BachelorMpcConfig config;
    config.prediction_dt_s = declare_parameter("prediction_dt_s", config.prediction_dt_s);
    config.weight_lateral_error = declare_parameter(
      "weight_lateral_error", config.weight_lateral_error);
    config.weight_heading_error = declare_parameter(
      "weight_heading_error", config.weight_heading_error);
    config.weight_velocity = declare_parameter("weight_velocity", config.weight_velocity);
    config.weight_lateral_velocity = declare_parameter(
      "weight_lateral_velocity", config.weight_lateral_velocity);
    config.weight_yaw_rate = declare_parameter("weight_yaw_rate", config.weight_yaw_rate);
    config.weight_steering_effort = declare_parameter(
      "weight_steering_effort", config.weight_steering_effort);
    config.weight_acceleration_effort = declare_parameter(
      "weight_acceleration_effort", config.weight_acceleration_effort);
    config.weight_steering_rate = declare_parameter(
      "weight_steering_rate", config.weight_steering_rate);
    config.weight_acceleration_rate = declare_parameter(
      "weight_acceleration_rate", config.weight_acceleration_rate);
    config.weight_effective_steering = declare_parameter(
      "weight_effective_steering", config.weight_effective_steering);
    config.cross_call_rate_scale = declare_parameter(
      "cross_call_rate_scale", config.cross_call_rate_scale);
    config.wall_margin = declare_parameter("wall_margin", config.wall_margin);
    config.max_solver_iterations = static_cast<std::size_t>(declare_parameter(
      "max_solver_iterations", static_cast<int64_t>(config.max_solver_iterations)));
    config.solver_convergence_tolerance = declare_parameter(
      "solver_convergence_tolerance", config.solver_convergence_tolerance);
    config.accept_max_iterations = declare_parameter(
      "accept_max_iterations", config.accept_max_iterations);

    std::string error;
    TrackModel track;
    if (!track.load_csv(trajectory_file_, &error)) {
      throw std::runtime_error(error);
    }
    controller_ = std::make_unique<BachelorMpcController>(track, config);

    diagnostics_pub_ = create_publisher<sdu_apex_msgs::msg::MpcDiagnostics>(
      diagnostics_topic_, 10);
    shadow_command_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      shadow_command_topic_, 10);
    state_sub_ = create_subscription<sdu_apex_msgs::msg::VehicleControlState>(
      state_topic_, rclcpp::QoS(20).reliable(),
      std::bind(&MpcShadowNode::state_callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "BachelorProject MPC shadow ready: model=%s trajectory=%s horizon=%d dt=%.3f; "
      "no actuator command is published",
      model_version_.c_str(), trajectory_file_.c_str(), PREDICTION_HORIZON,
      config.prediction_dt_s);
  }

private:
  void state_callback(const sdu_apex_msgs::msg::VehicleControlState::SharedPtr msg)
  {
    const auto start = std::chrono::steady_clock::now();
    sdu_apex_msgs::msg::MpcDiagnostics diagnostic;
    diagnostic.header = msg->header;
    diagnostic.shadow_only = true;
    diagnostic.model_version = model_version_;
    diagnostic.horizon_steps = PREDICTION_HORIZON;

    if (!msg->localization_valid) {
      diagnostic.failure_reason = "localization state invalid or stale";
      publish_diagnostics(diagnostic, start);
      return;
    }
    if (!reset_epoch_seen_ || msg->reset_epoch != reset_epoch_) {
      controller_->reset();
      reset_epoch_ = msg->reset_epoch;
      reset_epoch_seen_ = true;
    }

    ModelState state;
    state.x = msg->x_map_m;
    state.y = msg->y_map_m;
    state.yaw = msg->yaw_map_rad;
    state.u = std::max(0.0, msg->body_u_mps);
    state.v = msg->dynamics_state_valid ? msg->body_v_mps : 0.0;
    state.r = msg->yaw_rate_radps;
    state.steering = msg->steering_angle_rad;
    state.steering_valid = msg->steering_valid;

    const auto result = controller_->solve(state);
    diagnostic.valid = result.valid;
    diagnostic.solver_success = result.solver_success;
    diagnostic.proposed_steering_angle_rad = result.steering_rad;
    diagnostic.proposed_acceleration_mps2 = result.acceleration_mps2;
    diagnostic.current_lateral_error_m = result.current_lateral_error_m;
    diagnostic.current_heading_error_rad = result.current_heading_error_rad;
    diagnostic.predicted_lateral_error_m = result.predicted_lateral_error_m;
    diagnostic.predicted_distance_m = result.predicted_progress_m;
    diagnostic.horizon_steps = static_cast<uint32_t>(result.horizon_steps);
    diagnostic.solver_iterations = static_cast<uint32_t>(result.solver_iterations);
    diagnostic.solver_status = static_cast<uint32_t>(result.status);
    diagnostic.solver_primal_residual = result.primal_residual;
    diagnostic.solver_dual_residual = result.dual_residual;
    if (!result.valid) {
      diagnostic.failure_reason = result.status_text;
    }

    if (result.valid) {
      ackermann_msgs::msg::AckermannDriveStamped command;
      command.header = msg->header;
      command.drive.steering_angle = result.steering_rad;
      command.drive.speed = 0.0;
      command.drive.acceleration = result.acceleration_mps2;
      shadow_command_pub_->publish(command);
    }
    publish_diagnostics(diagnostic, start);
  }

  void publish_diagnostics(
    sdu_apex_msgs::msg::MpcDiagnostics &diagnostic,
    const std::chrono::steady_clock::time_point start)
  {
    diagnostic.solve_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
    diagnostics_pub_->publish(diagnostic);
  }

  std::string trajectory_file_;
  std::string state_topic_;
  std::string diagnostics_topic_;
  std::string shadow_command_topic_;
  std::string model_version_;
  std::uint64_t reset_epoch_{0};
  bool reset_epoch_seen_{false};
  std::unique_ptr<BachelorMpcController> controller_;
  rclcpp::Publisher<sdu_apex_msgs::msg::MpcDiagnostics>::SharedPtr diagnostics_pub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr shadow_command_pub_;
  rclcpp::Subscription<sdu_apex_msgs::msg::VehicleControlState>::SharedPtr state_sub_;
};

}  // namespace f1tenth_mpc

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<f1tenth_mpc::MpcShadowNode>());
  } catch (const std::exception &error) {
    fprintf(stderr, "mpc_shadow_node failed: %s\n", error.what());
  }
  rclcpp::shutdown();
  return 0;
}

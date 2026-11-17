#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sdu_apex_msgs/msg/unity_plant_state.hpp>
#include <sdu_apex_msgs/msg/drive_command.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "f1tenth_localization/bridge_epoch_assembler.hpp"
#include "f1tenth_localization/odometry_pipeline.hpp"
#include "f1tenth_localization/sensor_packet_assembler.hpp"

namespace
{

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

double roll_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.x + q.y * q.z),
    1.0 - 2.0 * (q.x * q.x + q.y * q.y));
}

double pitch_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::asin(std::clamp(
    2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0));
}

geometry_msgs::msg::Quaternion quaternion_from_yaw(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  const double half = 0.5 * yaw;
  q.z = std::sin(half);
  q.w = std::cos(half);
  return q;
}

int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL +
         static_cast<int64_t>(stamp.nanosec);
}

}  // namespace

class SensorOdometryNode final : public rclcpp::Node
{
public:
  SensorOdometryNode()
  : Node("sensor_odometry")
  {
    declare_parameter("left_encoder_topic", "/autodrive/roboracer_1/left_encoder");
    declare_parameter("right_encoder_topic", "/autodrive/roboracer_1/right_encoder");
    declare_parameter("imu_topic", "/autodrive/roboracer_1/imu");
    declare_parameter("bridge_request_command_topic", "/odom/bridge_request_command");
    declare_parameter("bridge_actuator_feedback_topic", "/odom/bridge_actuator_feedback");
    declare_parameter("odom_topic", "/odom");
    declare_parameter("diagnostics_topic", "/odom/diagnostics");
    declare_parameter("publish_diagnostics", false);
    declare_parameter("unity_plant_state_topic", "/odom/unity_plant_state");

    declare_parameter("odom_frame", "odom");
    declare_parameter("base_frame", "base_link");
    declare_parameter("publish_tf", true);
    declare_parameter("lidar_frame", "lidar");
    declare_parameter("imu_frame", "imu");
    declare_parameter("lidar_x_m", 0.2733);
    declare_parameter("lidar_y_m", 0.0);
    declare_parameter("lidar_z_m", 0.096);
    declare_parameter("imu_x_m", 0.08);
    declare_parameter("imu_y_m", 0.0);
    declare_parameter("imu_z_m", 0.055);
    declare_parameter("pose_xy_variance", 0.01);
    declare_parameter("pose_yaw_variance", 0.0001);
    declare_parameter("twist_linear_variance", 0.04);
    declare_parameter("twist_yaw_variance", 0.04);
    declare_parameter("wheel_radius_m", 0.059);
    declare_parameter("encoder_ticks_per_revolution", 1920.0);
    declare_parameter("imu_reference_forward_m", 0.15532);
    declare_parameter("maximum_packet_dt_s", 0.10);
    declare_parameter("lateral_acceleration_residual_gain", 1.0);

    observer_config_.wheel_radius_m = get_parameter("wheel_radius_m").as_double();
    observer_config_.encoder_ticks_per_revolution =
      get_parameter("encoder_ticks_per_revolution").as_double();
    observer_config_.imu_reference_forward_m =
      get_parameter("imu_reference_forward_m").as_double();
    observer_config_.maximum_packet_dt_s =
      get_parameter("maximum_packet_dt_s").as_double();
    observer_config_.lateral_acceleration_residual_gain =
      get_parameter("lateral_acceleration_residual_gain").as_double();
    pipeline_ = std::make_unique<f1tenth_localization::OdometryPipeline>(observer_config_);

    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    lidar_frame_ = get_parameter("lidar_frame").as_string();
    imu_frame_ = get_parameter("imu_frame").as_string();
    lidar_x_m_ = get_parameter("lidar_x_m").as_double();
    lidar_y_m_ = get_parameter("lidar_y_m").as_double();
    lidar_z_m_ = get_parameter("lidar_z_m").as_double();
    imu_x_m_ = get_parameter("imu_x_m").as_double();
    imu_y_m_ = get_parameter("imu_y_m").as_double();
    imu_z_m_ = get_parameter("imu_z_m").as_double();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    pose_xy_variance_ = std::max(0.0, get_parameter("pose_xy_variance").as_double());
    pose_yaw_variance_ = std::max(0.0, get_parameter("pose_yaw_variance").as_double());
    twist_linear_variance_ = std::max(
      0.0, get_parameter("twist_linear_variance").as_double());
    twist_yaw_variance_ = std::max(
      0.0, get_parameter("twist_yaw_variance").as_double());

    odom_message_.header.frame_id = odom_frame_;
    odom_message_.child_frame_id = base_frame_;
    plant_state_message_.header.frame_id = odom_frame_;
    transform_message_.header.frame_id = odom_frame_;
    transform_message_.child_frame_id = base_frame_;

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), rclcpp::QoS(100));
    if (get_parameter("publish_diagnostics").as_bool()) {
      diagnostics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("diagnostics_topic").as_string(), rclcpp::QoS(100));
      diagnostics_message_.layout.dim.resize(1);
      diagnostics_message_.layout.dim[0].label =
        "sensor_corrected_source_derived_odom_v3";
      diagnostics_message_.layout.dim[0].size = 42;
      diagnostics_message_.layout.dim[0].stride = 42;
      diagnostics_message_.data.resize(42);
    }
    plant_state_pub_ = create_publisher<sdu_apex_msgs::msg::UnityPlantState>(
      get_parameter("unity_plant_state_topic").as_string(),
      rclcpp::SensorDataQoS().keep_last(1));

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(100);
    left_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("left_encoder_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
        encoder_callback(*message, true);
      });
    right_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("right_encoder_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
        encoder_callback(*message, false);
      });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::Imu::ConstSharedPtr message) {
        imu_callback(*message);
      });
    bridge_request_command_sub_ =
      create_subscription<sdu_apex_msgs::msg::DriveCommand>(
      get_parameter("bridge_request_command_topic").as_string(), rclcpp::QoS(100),
      [this](sdu_apex_msgs::msg::DriveCommand::ConstSharedPtr message) {
        bridge_actuator_callback(*message, false);
      });
    bridge_actuator_feedback_sub_ =
      create_subscription<sdu_apex_msgs::msg::DriveCommand>(
      get_parameter("bridge_actuator_feedback_topic").as_string(), rclcpp::QoS(100),
      [this](sdu_apex_msgs::msg::DriveCommand::ConstSharedPtr message) {
        bridge_actuator_callback(*message, true);
      });

    packet_assembler_.set_packet_callback(
      [this](const f1tenth_localization::SensorPacket & packet) {
        try {
          bridge_epochs_.push_packet(packet);
          drain_bridge_epochs();
        } catch (const std::exception & error) {
          RCLCPP_FATAL(get_logger(), "odometry plant update failed: %s", error.what());
          rclcpp::shutdown();
        }
      });

    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
      static_tf_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
      publish_static_transforms();
    }
    RCLCPP_INFO(
      get_logger(),
      "In-process source-derived 1 kHz plant odometry active at a fixed 25 ms sensor epoch");
  }

private:
  void publish_static_transforms()
  {
    geometry_msgs::msg::TransformStamped lidar;
    lidar.header.stamp = now();
    lidar.header.frame_id = base_frame_;
    lidar.child_frame_id = lidar_frame_;
    lidar.transform.translation.x = lidar_x_m_;
    lidar.transform.translation.y = lidar_y_m_;
    lidar.transform.translation.z = lidar_z_m_;
    lidar.transform.rotation.w = 1.0;
    geometry_msgs::msg::TransformStamped imu;
    imu.header = lidar.header;
    imu.header.frame_id = base_frame_;
    imu.child_frame_id = imu_frame_;
    imu.transform.translation.x = imu_x_m_;
    imu.transform.translation.y = imu_y_m_;
    imu.transform.translation.z = imu_z_m_;
    imu.transform.rotation.w = 1.0;
    static_tf_broadcaster_->sendTransform({lidar, imu});
  }

  void imu_callback(const sensor_msgs::msg::Imu & message)
  {
    const double yaw = yaw_from_quaternion(message.orientation);
    const double roll = roll_from_quaternion(message.orientation);
    const double pitch = pitch_from_quaternion(message.orientation);
    if (!std::isfinite(yaw) || !std::isfinite(message.linear_acceleration.x) ||
      !std::isfinite(message.linear_acceleration.y) ||
      !std::isfinite(message.angular_velocity.x) ||
      !std::isfinite(message.angular_velocity.y) ||
      !std::isfinite(message.angular_velocity.z) ||
      !std::isfinite(roll) || !std::isfinite(pitch))
    {
      return;
    }
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    try {
    packet_assembler_.add_imu_sample(
      stamp_ns(message.header.stamp), message.linear_acceleration.x,
      message.linear_acceleration.y, message.angular_velocity.z, yaw,
      roll, pitch, message.angular_velocity.x, message.angular_velocity.y);
    } catch (const std::exception & error) {
      RCLCPP_FATAL(get_logger(), "sensor epoch assembly failed: %s", error.what());
      rclcpp::shutdown();
    }
  }

  void encoder_callback(const sensor_msgs::msg::JointState & message, bool left)
  {
    if (message.position.empty() || !std::isfinite(message.position.front())) {
      return;
    }
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    try {
    packet_assembler_.add_encoder_sample(
      stamp_ns(message.header.stamp), message.position.front(), left);
    } catch (const std::exception & error) {
      RCLCPP_FATAL(get_logger(), "sensor epoch assembly failed: %s", error.what());
      rclcpp::shutdown();
    }
  }

  void bridge_actuator_callback(
    const sdu_apex_msgs::msg::DriveCommand & message, bool feedback)
  {
    constexpr double kMaxSteeringRad = 3.14159265358979323846 / 6.0;
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    try {
      const auto stamp = stamp_ns(message.header.stamp);
      const double steering = message.steering_angle_rad / kMaxSteeringRad;
      const bool valid = feedback ? bridge_epochs_.push_feedback(
        stamp, steering, message.throttle_normalized) : bridge_epochs_.push_command(
        stamp, steering, message.throttle_normalized);
      if (!valid) throw std::runtime_error("invalid stamped Bridge actuator sample");
      drain_bridge_epochs();
    } catch (const std::exception & error) {
      RCLCPP_FATAL(get_logger(), "odometry epoch join failed: %s", error.what());
      rclcpp::shutdown();
    }
  }

  void drain_bridge_epochs()
  {
    // sensor_mutex_ is held by both the sensor assembler and actuator callbacks.
    f1tenth_localization::SensorPacket packet;
    f1tenth_localization::BridgeEpochAssembler::Command command, feedback;
    while (bridge_epochs_.pop_ready(&packet, &command, &feedback))
      process_packet(packet, command, feedback);
  }

  void process_packet(
    const f1tenth_localization::SensorPacket & packet,
    const f1tenth_localization::BridgeEpochAssembler::Command & command,
    const f1tenth_localization::BridgeEpochAssembler::Command & measured_feedback)
  {
    const auto packet_start = diagnostics_pub_
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
    constexpr double kMaxSteeringRad = 3.14159265358979323846 / 6.0;
    const double steering_command = command.steering_normalized;
    const double throttle_command = command.throttle_normalized;
    const double steering_feedback = measured_feedback.steering_normalized * kMaxSteeringRad;
    const double throttle_feedback = measured_feedback.throttle_normalized;
    constexpr bool feedback_valid = true;

    const auto step = pipeline_->step(
      packet, steering_command, throttle_command, steering_feedback, throttle_feedback);
    publish(step.estimate, packet, steering_command, throttle_command,
      steering_feedback, throttle_feedback, feedback_valid, packet_start,
      step.plant_advance_us, step.observer_update_us, step.plant_correction_us,
      step.raw_plant);
  }

  void publish(
    const f1tenth_localization::OdometryEstimate & estimate,
    const f1tenth_localization::SensorPacket & packet,
    double steering_command, double throttle_command,
    double steering_feedback, double throttle_feedback, bool feedback_valid,
    std::chrono::steady_clock::time_point packet_start,
    double plant_advance_us, double observer_update_us,
    double plant_correction_us,
    const f1tenth_localization::PredictedPlantState & raw_plant)
  {
    const auto publish_start = diagnostics_pub_
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
    // All three sensor streams were assembled by their source timestamp. Keep
    // the published odometry and plant state on that same epoch; publication
    // time here is later and can include packet-reordering/dispatch latency.
    const rclcpp::Time stamp(packet.stamp_ns, get_clock()->get_clock_type());
    auto & odom = odom_message_;
    odom.header.stamp = stamp;
    odom.pose.pose.position.x = estimate.x_m;
    odom.pose.pose.position.y = estimate.y_m;
    odom.pose.pose.orientation = quaternion_from_yaw(estimate.yaw_rad);
    odom.twist.twist.linear.x = estimate.body_u_mps;
    odom.twist.twist.linear.y = estimate.body_v_mps;
    odom.twist.twist.angular.z = estimate.yaw_rate_radps;
    odom.pose.covariance[0] = pose_xy_variance_;
    odom.pose.covariance[7] = pose_xy_variance_;
    odom.pose.covariance[35] = pose_yaw_variance_;
    odom.twist.covariance[0] = twist_linear_variance_;
    odom.twist.covariance[7] = twist_linear_variance_;
    odom.twist.covariance[35] = twist_yaw_variance_;

    auto & state = plant_state_message_;
    state.header.stamp = stamp;
    state.state_schema_version = 11;
    state.state_origin = 1;
    state.suspension_history_initialized = pipeline_->plant().suspension_history_initialized;
    state.sensor_corrected_state_mask =
      (1u << 0) | (1u << 1) | (1u << 2) | (1u << 5) | (1u << 6) |
      (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12) |
      (feedback_valid ? (1u << 13) : 0u);
    state.longitudinal_velocity_mps = estimate.body_u_mps;
    state.lateral_velocity_com_mps = estimate.body_v_com_mps;
    state.lateral_velocity_rear_axle_mps = estimate.body_v_mps;
    state.physx_model_rear_axle_x_m = estimate.physx_model_x_m;
    state.physx_model_rear_axle_y_m = estimate.physx_model_y_m;
    state.yaw_rate_radps = estimate.yaw_rate_radps;
    state.gyro_yaw_rate_radps = packet.yaw_rate_radps;
    state.yaw_rate_prediction_residual_valid =
      pipeline_->runtime().yaw_rate_prediction_residual_valid();
    state.yaw_rate_prediction_residual_radps =
      state.yaw_rate_prediction_residual_valid ?
      pipeline_->runtime().yaw_rate_prediction_residual_radps() : 0.0;
    state.steering_command_normalized = steering_command;
    state.throttle_command_normalized = throttle_command;
    state.delayed_steering_command_1_normalized =
      pipeline_->plant().delayed_steering_commands_normalized[0];
    state.delayed_steering_command_2_normalized =
      pipeline_->plant().delayed_steering_commands_normalized[1];
    state.delayed_throttle_command_1_normalized =
      pipeline_->plant().delayed_throttle_commands_normalized[0];
    state.delayed_throttle_command_2_normalized =
      pipeline_->plant().delayed_throttle_commands_normalized[1];
    state.actuator_feedback_valid = feedback_valid;
    state.steering_feedback_rad = steering_feedback;
    state.throttle_feedback_normalized = throttle_feedback;
    std::copy(pipeline_->plant().state.begin() + 3, pipeline_->plant().state.begin() + 7,
      state.wheel_omega_radps.begin());
    state.wheel_rotation_angle_rad = pipeline_->plant().wheel_rotation_angle_rad;
    state.body_height_m = pipeline_->plant().state[7];
    state.body_height_rate_mps = pipeline_->plant().state[8];
    state.roll_rad = pipeline_->plant().state[9];
    state.roll_rate_radps = pipeline_->plant().state[10];
    state.pitch_rad = pipeline_->plant().state[11];
    state.pitch_rate_radps = pipeline_->plant().state[12];
    state.physical_steering_rad = pipeline_->plant().state[13];
    std::copy_n(pipeline_->plant().state.begin() + 14, 4,
      state.suspension_jounce_m.begin());
    std::copy_n(pipeline_->plant().state.begin() +
          UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL, 4,
      state.low_forward_speed_time_s.begin());
    std::copy_n(pipeline_->plant().state.begin() +
          UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL, 4,
      state.low_side_speed_time_s.begin());
    state.longitudinal_slip = pipeline_->plant().longitudinal_slip;
    state.lateral_slip = pipeline_->plant().lateral_slip;
    state.normal_load_n = pipeline_->plant().normal_load_n;
    state.suspension_force_n = pipeline_->plant().suspension_force_n;
    state.suspension_jounce_rate_mps = pipeline_->plant().suspension_jounce_rate_mps;
    state.tire_longitudinal_force_n = pipeline_->plant().tire_longitudinal_force_n;
    state.tire_lateral_force_n = pipeline_->plant().tire_lateral_force_n;
    state.longitudinal_acceleration_mps2 = packet.ax_mps2;
    state.lateral_acceleration_mps2 = packet.ay_mps2;
    state.yaw_acceleration_radps2 = pipeline_->plant().yaw_acceleration_radps2;
    // Publish the hidden state before odometry triggers the MPC callback.
    // DDS does not guarantee order across topics, so the MPC also accepts a
    // bounded causal one-packet-old snapshot and propagates it to this epoch.
    plant_state_pub_->publish(state);
    odom_pub_->publish(odom);

    if (publish_tf_) {
      auto & transform = transform_message_;
      transform.header.stamp = stamp;
      transform.transform.translation.x = estimate.x_m;
      transform.transform.translation.y = estimate.y_m;
      transform.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(transform);
    }

    if (diagnostics_pub_) {
      const double publish_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - publish_start).count();
      const double process_packet_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - packet_start).count();
      auto & diagnostics = diagnostics_message_.data;
      diagnostics = {
        estimate.stamp_s, estimate.dt_s, estimate.body_u_mps,
        estimate.body_v_mps, estimate.body_v_com_mps,
        estimate.yaw_rate_radps, estimate.speed_mps,
        estimate.encoder_rear_axle_speed_mps,
        packet.ax_mps2, packet.ay_mps2,
        raw_plant.state[0], raw_plant.state[1],
        raw_plant.raw_yaw_rate_prediction_radps,
        steering_command, throttle_command,
        static_cast<double>(packet_assembler_.packet_drop_count()),
        static_cast<double>(packet_assembler_.packet_coherence_fault_count()),
        plant_advance_us, observer_update_us, plant_correction_us,
        publish_us, process_packet_us,
        raw_plant.state[3 + 2], raw_plant.state[3 + 3],
        estimate.encoder_wheel_omega_average_radps[0],
        estimate.encoder_wheel_omega_average_radps[1],
        estimate.encoder_model_rotation_residual_rad[0],
        estimate.encoder_model_rotation_residual_rad[1],
        estimate.wheel_omega_correction_radps[0],
        estimate.wheel_omega_correction_radps[1],
        estimate.physx_model_x_m, estimate.physx_model_y_m,
        raw_plant.longitudinal_acceleration_mps2,
        raw_plant.lateral_acceleration_mps2,
        estimate.lateral_acceleration_residual_gain,
        pipeline_->plant().wheel_encoder_angle_prediction_rad[0],
        pipeline_->plant().wheel_encoder_angle_prediction_rad[1],
        pipeline_->plant().wheel_encoder_angle_prediction_rad[2],
        pipeline_->plant().wheel_encoder_angle_prediction_rad[3],
        raw_plant.corrected_yaw_rate_prediction_radps,
        raw_plant.yaw_rate_prediction_correction_radps,
        static_cast<double>(raw_plant.yaw_rate_prediction_mode)};
      diagnostics_pub_->publish(diagnostics_message_);
    }
  }

  f1tenth_localization::OdometryObserverConfig observer_config_{};
  f1tenth_localization::SensorPacketAssembler packet_assembler_;
  std::unique_ptr<f1tenth_localization::OdometryPipeline> pipeline_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
  std_msgs::msg::Float64MultiArray diagnostics_message_;
  rclcpp::Publisher<sdu_apex_msgs::msg::UnityPlantState>::SharedPtr plant_state_pub_;
  nav_msgs::msg::Odometry odom_message_;
  sdu_apex_msgs::msg::UnityPlantState plant_state_message_;
  geometry_msgs::msg::TransformStamped transform_message_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sdu_apex_msgs::msg::DriveCommand>::SharedPtr bridge_request_command_sub_;
  rclcpp::Subscription<sdu_apex_msgs::msg::DriveCommand>::SharedPtr bridge_actuator_feedback_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::mutex sensor_mutex_;
  f1tenth_localization::BridgeEpochAssembler bridge_epochs_;

  std::string odom_frame_{"odom"};
  std::string base_frame_{"base_link"};
  std::string lidar_frame_{"lidar"};
  std::string imu_frame_{"imu"};
  double lidar_x_m_{0.2733};
  double lidar_y_m_{0.0};
  double lidar_z_m_{0.096};
  double imu_x_m_{0.08};
  double imu_y_m_{0.0};
  double imu_z_m_{0.055};
  double pose_xy_variance_{0.01};
  double pose_yaw_variance_{0.0001};
  double twist_linear_variance_{0.04};
  double twist_yaw_variance_{0.04};
  bool publish_tf_{true};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorOdometryNode>());
  rclcpp::shutdown();
  return 0;
}

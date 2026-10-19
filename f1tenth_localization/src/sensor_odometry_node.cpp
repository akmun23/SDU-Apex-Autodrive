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
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "f1tenth_localization/odometry_observer.hpp"
#include "f1tenth_localization/physx_odom_runtime.hpp"
#include "f1tenth_localization/sensor_packet_assembler.hpp"

namespace
{

constexpr double kPhysicsPacketPeriodS = 0.025;
constexpr auto kActuatorFeedbackMaxAge = std::chrono::milliseconds(100);

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
    declare_parameter("steering_feedback_topic", "/autodrive/roboracer_1/steering");
    declare_parameter("throttle_feedback_topic", "/autodrive/roboracer_1/throttle");
    declare_parameter("steering_command_topic", "/autodrive/roboracer_1/steering_command");
    declare_parameter("throttle_command_topic", "/autodrive/roboracer_1/throttle_command");
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
    observer_ = f1tenth_localization::OdometryObserver(observer_config_);

    plant_runtime_ = std::make_unique<f1tenth_localization::SourceDerivedOdomRuntime>(
      observer_config_.imu_reference_forward_m);

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
    steering_feedback_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("steering_feedback_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr message) {
        actuator_feedback_callback(*message, true);
      });
    throttle_feedback_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("throttle_feedback_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr message) {
        actuator_feedback_callback(*message, false);
      });
    steering_command_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("steering_command_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr message) {
        actuator_command_callback(*message, true);
      });
    throttle_command_sub_ = create_subscription<std_msgs::msg::Float32>(
      get_parameter("throttle_command_topic").as_string(), sensor_qos,
      [this](std_msgs::msg::Float32::ConstSharedPtr message) {
        actuator_command_callback(*message, false);
      });

    packet_assembler_.set_packet_callback(
      [this](const f1tenth_localization::SensorPacket & packet) {
        try {
          process_packet(packet);
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
    packet_assembler_.add_imu_sample(
      stamp_ns(message.header.stamp), message.linear_acceleration.x,
      message.linear_acceleration.y, message.angular_velocity.z, yaw,
      roll, pitch, message.angular_velocity.x, message.angular_velocity.y);
  }

  void encoder_callback(const sensor_msgs::msg::JointState & message, bool left)
  {
    if (message.position.empty() || !std::isfinite(message.position.front())) {
      return;
    }
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    packet_assembler_.add_encoder_sample(
      stamp_ns(message.header.stamp), message.position.front(), left);
  }

  void actuator_feedback_callback(const std_msgs::msg::Float32 & message, bool steering)
  {
    const double value = static_cast<double>(message.data);
    if (!std::isfinite(value)) {
      return;
    }
    std::lock_guard<std::mutex> lock(actuator_mutex_);
    const auto received = std::chrono::steady_clock::now();
    if (steering) {
      steering_feedback_rad_ = value;
      steering_feedback_received_ = received;
      have_steering_feedback_ = true;
    } else {
      throttle_feedback_ = value;
      throttle_feedback_received_ = received;
      have_throttle_feedback_ = true;
    }
  }

  void actuator_command_callback(const std_msgs::msg::Float32 & message, bool steering)
  {
    const double value = static_cast<double>(message.data);
    if (!std::isfinite(value)) {
      return;
    }
    std::lock_guard<std::mutex> lock(actuator_mutex_);
    if (steering) {
      steering_command_norm_ = value;
    } else {
      throttle_command_norm_ = value;
    }
  }

  void process_packet(const f1tenth_localization::SensorPacket & packet)
  {
    const auto packet_start = diagnostics_pub_
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
    if (!clock_initialized_) {
      clock_initialized_ = true;
      observer_stamp_s_ = 0.0;
    } else {
      observer_stamp_s_ += kPhysicsPacketPeriodS;
    }

    double steering_command = 0.0;
    double throttle_command = 0.0;
    double steering_feedback = 0.0;
    double throttle_feedback = 0.0;
    bool feedback_valid = false;
    {
      std::lock_guard<std::mutex> lock(actuator_mutex_);
      steering_command = steering_command_norm_;
      throttle_command = throttle_command_norm_;
      const auto current = std::chrono::steady_clock::now();
      feedback_valid = have_steering_feedback_ && have_throttle_feedback_ &&
        current - steering_feedback_received_ <= kActuatorFeedbackMaxAge &&
        current - throttle_feedback_received_ <= kActuatorFeedbackMaxAge;
      if (feedback_valid) {
        steering_feedback = steering_feedback_rad_;
        throttle_feedback = throttle_feedback_;
      }
    }

    std::chrono::steady_clock::time_point plant_advance_start{};
    if (diagnostics_pub_)
      plant_advance_start = std::chrono::steady_clock::now();
    last_plant_ = plant_runtime_->advance(
      steering_command, throttle_command, feedback_valid,
      steering_feedback);
    const double plant_advance_us = diagnostics_pub_
      ? std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - plant_advance_start).count()
      : 0.0;

    f1tenth_localization::OdometryObservation observation;
    observation.stamp_s = observer_stamp_s_;
    observation.left_angle_rad = packet.left_angle_rad;
    observation.right_angle_rad = packet.right_angle_rad;
    observation.ax_mps2 = packet.ax_mps2;
    observation.ay_mps2 = packet.ay_mps2;
    observation.yaw_rate_radps = packet.yaw_rate_radps;
    observation.yaw_rad = packet.imu_yaw_rad;  // Only the initial heading uses orientation.
    observation.roll_rad = packet.roll_rad;
    observation.pitch_rad = packet.pitch_rad;
    observation.roll_rate_radps = packet.roll_rate_radps;
    observation.pitch_rate_radps = packet.pitch_rate_radps;
    observation.steering_feedback_rad = steering_feedback;
    observation.throttle_feedback = throttle_feedback;
    observation.actuator_feedback_valid = feedback_valid;
    observation.steering_command_normalized = steering_command;
    observation.throttle_command_normalized = throttle_command;
    observation.actuator_command_valid = true;
    observation.physx_forward_velocity_mps = last_plant_.state[0];
    observation.physx_lateral_velocity_com_mps = last_plant_.state[1];
    observation.physx_yaw_rate_radps = last_plant_.state[2];
    observation.physx_longitudinal_acceleration_mps2 =
      last_plant_.longitudinal_acceleration_mps2;
    observation.physx_lateral_acceleration_mps2 =
      last_plant_.lateral_acceleration_mps2;
    observation.physx_rear_axle_x_m = last_plant_.rear_axle_x_m;
    observation.physx_rear_axle_y_m = last_plant_.rear_axle_y_m;
    observation.physx_pose_valid = true;
    std::copy(last_plant_.state.begin() + 3, last_plant_.state.begin() + 7,
      observation.physx_wheel_omega_radps.begin());
    observation.physx_wheel_rotation_angle_rad =
      last_plant_.wheel_rotation_angle_rad;
    observation.physx_wheel_encoder_angle_prediction_rad =
      last_plant_.wheel_encoder_angle_prediction_rad;
    observation.physx_state = last_plant_.state;
    observation.physx_longitudinal_slip = last_plant_.longitudinal_slip;
    observation.physx_lateral_slip = last_plant_.lateral_slip;
    observation.physx_normal_load_n = last_plant_.normal_load_n;
    observation.physx_state_valid = true;

    f1tenth_localization::PredictedPlantState raw_plant{};
    if (diagnostics_pub_)
      raw_plant = last_plant_;
    std::chrono::steady_clock::time_point observer_update_start{};
    if (diagnostics_pub_)
      observer_update_start = std::chrono::steady_clock::now();
    const auto estimate = observer_.update(observation);
    const double observer_update_us = diagnostics_pub_
      ? std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - observer_update_start).count()
      : 0.0;
    std::chrono::steady_clock::time_point plant_correction_start{};
    if (diagnostics_pub_)
      plant_correction_start = std::chrono::steady_clock::now();
    plant_runtime_->correct(
      estimate.body_u_mps, estimate.body_v_com_mps,
      estimate.yaw_rate_radps, estimate.yaw_rad,
      estimate.roll_rad, estimate.pitch_rad,
      estimate.unity_state[8],
      estimate.roll_rate_radps, estimate.pitch_rate_radps,
      estimate.wheel_omega_radps);
    const double plant_correction_us = diagnostics_pub_
      ? std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - plant_correction_start).count()
      : 0.0;
    last_plant_.state = estimate.unity_state;
    last_plant_.longitudinal_acceleration_mps2 =
      estimate.longitudinal_acceleration_mps2;
    last_plant_.lateral_acceleration_mps2 = estimate.lateral_acceleration_mps2;
    publish(estimate, packet, steering_command, throttle_command,
      steering_feedback, throttle_feedback, feedback_valid, packet_start,
      plant_advance_us, observer_update_us, plant_correction_us, raw_plant);
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
    state.suspension_history_initialized = last_plant_.suspension_history_initialized;
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
      plant_runtime_->yaw_rate_prediction_residual_valid();
    state.yaw_rate_prediction_residual_radps =
      state.yaw_rate_prediction_residual_valid ?
      plant_runtime_->yaw_rate_prediction_residual_radps() : 0.0;
    state.steering_command_normalized = steering_command;
    state.throttle_command_normalized = throttle_command;
    state.delayed_steering_command_1_normalized =
      last_plant_.delayed_steering_commands_normalized[0];
    state.delayed_steering_command_2_normalized =
      last_plant_.delayed_steering_commands_normalized[1];
    state.delayed_throttle_command_1_normalized =
      last_plant_.delayed_throttle_commands_normalized[0];
    state.delayed_throttle_command_2_normalized =
      last_plant_.delayed_throttle_commands_normalized[1];
    state.actuator_feedback_valid = feedback_valid;
    state.steering_feedback_rad = steering_feedback;
    state.throttle_feedback_normalized = throttle_feedback;
    std::copy(last_plant_.state.begin() + 3, last_plant_.state.begin() + 7,
      state.wheel_omega_radps.begin());
    state.wheel_rotation_angle_rad = last_plant_.wheel_rotation_angle_rad;
    state.body_height_m = last_plant_.state[7];
    state.body_height_rate_mps = last_plant_.state[8];
    state.roll_rad = last_plant_.state[9];
    state.roll_rate_radps = last_plant_.state[10];
    state.pitch_rad = last_plant_.state[11];
    state.pitch_rate_radps = last_plant_.state[12];
    state.physical_steering_rad = feedback_valid ?
      steering_feedback : last_plant_.state[13];
    std::copy_n(last_plant_.state.begin() + 14, 4,
      state.suspension_jounce_m.begin());
    std::copy_n(last_plant_.state.begin() +
          UNITY_SOURCE_LOW_FORWARD_SPEED_TIME_FL, 4,
      state.low_forward_speed_time_s.begin());
    std::copy_n(last_plant_.state.begin() +
          UNITY_SOURCE_LOW_SIDE_SPEED_TIME_FL, 4,
      state.low_side_speed_time_s.begin());
    state.longitudinal_slip = last_plant_.longitudinal_slip;
    state.lateral_slip = last_plant_.lateral_slip;
    state.normal_load_n = last_plant_.normal_load_n;
    state.suspension_force_n = last_plant_.suspension_force_n;
    state.suspension_jounce_rate_mps = last_plant_.suspension_jounce_rate_mps;
    state.tire_longitudinal_force_n = last_plant_.tire_longitudinal_force_n;
    state.tire_lateral_force_n = last_plant_.tire_lateral_force_n;
    state.longitudinal_acceleration_mps2 = packet.ax_mps2;
    state.lateral_acceleration_mps2 = packet.ay_mps2;
    state.yaw_acceleration_radps2 = last_plant_.yaw_acceleration_radps2;
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
        last_plant_.wheel_encoder_angle_prediction_rad[0],
        last_plant_.wheel_encoder_angle_prediction_rad[1],
        last_plant_.wheel_encoder_angle_prediction_rad[2],
        last_plant_.wheel_encoder_angle_prediction_rad[3],
        raw_plant.corrected_yaw_rate_prediction_radps,
        raw_plant.yaw_rate_prediction_correction_radps,
        static_cast<double>(raw_plant.yaw_rate_prediction_mode)};
      diagnostics_pub_->publish(diagnostics_message_);
    }
  }

  f1tenth_localization::OdometryObserverConfig observer_config_{};
  f1tenth_localization::OdometryObserver observer_{};
  f1tenth_localization::SensorPacketAssembler packet_assembler_;
  std::unique_ptr<f1tenth_localization::SourceDerivedOdomRuntime> plant_runtime_;
  f1tenth_localization::PredictedPlantState last_plant_{};
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
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steering_feedback_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr throttle_feedback_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steering_command_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr throttle_command_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::mutex sensor_mutex_;
  std::mutex actuator_mutex_;
  bool clock_initialized_{false};
  double observer_stamp_s_{0.0};
  double steering_command_norm_{0.0};
  double throttle_command_norm_{0.0};
  double steering_feedback_rad_{0.0};
  double throttle_feedback_{0.0};
  std::chrono::steady_clock::time_point steering_feedback_received_{};
  std::chrono::steady_clock::time_point throttle_feedback_received_{};
  bool have_steering_feedback_{false};
  bool have_throttle_feedback_{false};

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

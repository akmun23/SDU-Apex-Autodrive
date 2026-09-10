#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "f1tenth_localization/odometry_observer.hpp"
#include "f1tenth_localization/sensor_packet_assembler.hpp"

namespace
{

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
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
  : Node("sensor_odometry"),
    observer_(default_observer_config()),
    packet_assembler_(8),
    tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
    static_tf_broadcaster_(
      std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this))
  {
    declare_parameter("left_encoder_topic", "/autodrive/roboracer_1/left_encoder");
    declare_parameter("right_encoder_topic", "/autodrive/roboracer_1/right_encoder");
    declare_parameter("imu_topic", "/autodrive/roboracer_1/imu");
    declare_parameter("odom_topic", "/odom");
    declare_parameter("diagnostics_topic", "/odom/diagnostics");
    declare_parameter("reset_enabled", false);
    declare_parameter("reset_topic", "/autodrive/reset_command");

    declare_parameter("odom_frame", "odom");
    declare_parameter("base_frame", "base_link");
    declare_parameter("lidar_frame", "lidar");
    declare_parameter("imu_frame", "imu");
    declare_parameter("lidar_x_m", 0.2733);
    declare_parameter("lidar_y_m", 0.0);
    declare_parameter("lidar_z_m", 0.096);
    declare_parameter("imu_x_m", 0.08);
    declare_parameter("imu_y_m", 0.0);
    declare_parameter("imu_z_m", 0.055);
    declare_parameter("imu_orientation_correction_gain", 1.0);
    declare_parameter("max_imu_orientation_step_rad", 0.30);
    declare_parameter("max_pending_packets", 8);
    declare_parameter("pose_xy_variance", 0.01);
    // An isolated bridge gap is diagnostic evidence, not a permanent loss of
    // localization. Keep its covariance conservative but below the controller
    // stop gate; the EKF treats source covariance as a floor for the epoch.
    declare_parameter("timing_degraded_pose_xy_variance", 0.01);
    declare_parameter("pose_yaw_variance", 0.0001);
    declare_parameter("twist_linear_variance", 0.04);
    declare_parameter("twist_yaw_variance", 0.04);

    declare_parameter("wheel_radius_m", 0.059);
    declare_parameter("wheel_speed_scale", 0.968);
    declare_parameter("reset_encoder_jump_rad", 50.0);
    declare_parameter("wheel_speed_window_s", 0.10);
    declare_parameter("normal_packet_dt_max_s", 0.080);
    declare_parameter("degraded_packet_dt_max_s", 0.100);
    declare_parameter("max_integratable_gap_s", 0.250);
    declare_parameter("decel_detect_ax_mps2", -0.5);
    declare_parameter("decel_ax_scale", 1.005);
    declare_parameter("decel_ax_offset_mps2", 0.020);
    declare_parameter("wheel_update_ax_abs_max_mps2", 6.5);
    declare_parameter("wheel_freeze_speed_mps", 0.15);
    declare_parameter("wheel_innovation_max_mps", 1.50);
    declare_parameter("stationary_speed_threshold_mps", 0.03);
    declare_parameter("wheel_burst_disagreement_mps", 1.0);
    declare_parameter("wheel_update_beta", 0.85);
    declare_parameter("stationary_hold_s", 0.10);
    declare_parameter("stationary_ax_abs_max_mps2", 0.25);
    declare_parameter("stationary_ay_abs_max_mps2", 0.75);
    declare_parameter("stationary_yaw_rate_abs_max_radps", 0.15);
    declare_parameter("turn_enter_yaw_rate_radps", 0.6);
    declare_parameter("turn_enter_abs_ay_mps2", 6.0);
    declare_parameter("turn_exit_yaw_rate_radps", 0.1);
    declare_parameter("turn_exit_abs_ay_mps2", 0.5);
    declare_parameter("turn_exit_hold_s", 0.5);
    declare_parameter("turn_wheel_braking_ax_mps2", -1.0);
    declare_parameter("integrate_lateral_acceleration_in_turn", false);
    declare_parameter("max_imu_ax_abs_mps2", 30.0);

    observer_config_ = load_observer_config();
    observer_ = f1tenth_localization::OdometryObserver(observer_config_);
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
    imu_orientation_correction_gain_ = std::clamp(
      get_parameter("imu_orientation_correction_gain").as_double(), 0.0, 1.0);
    max_imu_orientation_step_rad_ = std::max(
      0.05, get_parameter("max_imu_orientation_step_rad").as_double());
    max_pending_packets_ = static_cast<std::size_t>(std::clamp<int64_t>(
      get_parameter("max_pending_packets").as_int(), 2, 32));
    packet_assembler_.set_max_pending_packets(max_pending_packets_);
    packet_assembler_.set_packet_callback(
      [this](const f1tenth_localization::SensorPacket & packet) {
        process_packet(packet);
      });
    pose_xy_variance_ = std::max(0.0, get_parameter("pose_xy_variance").as_double());
    timing_degraded_pose_xy_variance_ = std::max(
      pose_xy_variance_, get_parameter("timing_degraded_pose_xy_variance").as_double());
    pose_yaw_variance_ = std::max(0.0, get_parameter("pose_yaw_variance").as_double());
    twist_linear_variance_ = std::max(
      0.0, get_parameter("twist_linear_variance").as_double());
    twist_yaw_variance_ = std::max(
      0.0, get_parameter("twist_yaw_variance").as_double());

    const auto derived_qos = rclcpp::QoS(100);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), derived_qos);
    diagnostics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      get_parameter("diagnostics_topic").as_string(), derived_qos);

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(100);
    left_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("left_encoder_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
        encoder_callback(*msg, true);
      });
    right_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("right_encoder_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
        encoder_callback(*msg, false);
      });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), sensor_qos,
      [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
        imu_callback(*msg);
      });
    if (get_parameter("reset_enabled").as_bool()) {
      reset_sub_ = create_subscription<std_msgs::msg::Bool>(
        get_parameter("reset_topic").as_string(), rclcpp::QoS(10),
        [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
          if (msg->data) {
            reset_diagnostic_epoch();
          }
        });
    }

    publish_static_transforms();
    RCLCPP_INFO(
      get_logger(),
      "Deterministic odometry observer: exact timestamp packets, encoders + IMU only");
  }

private:
  static f1tenth_localization::OdometryObserverConfig default_observer_config()
  {
    f1tenth_localization::OdometryObserverConfig config;
    config.wheel_radius_m = 0.059;
    config.wheel_speed_scale = 0.968;
    config.reset_encoder_jump_rad = 50.0;
    config.wheel_speed_window_s = 0.10;
    config.normal_packet_dt_max_s = 0.080;
    config.degraded_packet_dt_max_s = 0.100;
    config.decel_detect_ax_mps2 = -0.5;
    config.decel_ax_scale = 1.005;
    config.decel_ax_offset_mps2 = 0.020;
    config.wheel_update_ax_abs_max_mps2 = 6.5;
    config.wheel_freeze_speed_mps = 0.15;
    config.wheel_innovation_max_mps = 1.50;
    config.wheel_burst_disagreement_mps = 1.0;
    config.stationary_speed_threshold_mps = 0.03;
    config.wheel_update_beta = 0.85;
    config.stationary_hold_s = 0.10;
    config.stationary_ax_abs_max_mps2 = 0.25;
    config.stationary_ay_abs_max_mps2 = 0.75;
    config.stationary_yaw_rate_abs_max_radps = 0.15;
    config.turn_enter_yaw_rate_radps = 0.6;
    config.turn_enter_abs_ay_mps2 = 6.0;
    config.turn_exit_yaw_rate_radps = 0.1;
    config.turn_exit_abs_ay_mps2 = 0.5;
    config.turn_exit_hold_s = 0.5;
    config.turn_wheel_braking_ax_mps2 = -1.0;
    config.imu_x_offset_m = 0.08;
    config.integrate_lateral_acceleration_in_turn = false;
    config.max_imu_ax_abs_mps2 = 30.0;
    return config;
  }

  f1tenth_localization::OdometryObserverConfig load_observer_config() const
  {
    auto config = default_observer_config();
    config.wheel_radius_m = get_parameter("wheel_radius_m").as_double();
    config.wheel_speed_scale = std::max(
      0.0, get_parameter("wheel_speed_scale").as_double());
    config.reset_encoder_jump_rad = get_parameter("reset_encoder_jump_rad").as_double();
    config.wheel_speed_window_s = std::max(
      0.0, get_parameter("wheel_speed_window_s").as_double());
    config.normal_packet_dt_max_s = get_parameter("normal_packet_dt_max_s").as_double();
    config.degraded_packet_dt_max_s = get_parameter("degraded_packet_dt_max_s").as_double();
    config.max_integratable_gap_s = get_parameter("max_integratable_gap_s").as_double();
    config.decel_detect_ax_mps2 = get_parameter("decel_detect_ax_mps2").as_double();
    config.decel_ax_scale = get_parameter("decel_ax_scale").as_double();
    config.decel_ax_offset_mps2 = get_parameter("decel_ax_offset_mps2").as_double();
    config.wheel_update_ax_abs_max_mps2 = get_parameter(
      "wheel_update_ax_abs_max_mps2").as_double();
    config.wheel_freeze_speed_mps = get_parameter("wheel_freeze_speed_mps").as_double();
    config.wheel_innovation_max_mps = get_parameter(
      "wheel_innovation_max_mps").as_double();
    config.wheel_burst_disagreement_mps = get_parameter(
      "wheel_burst_disagreement_mps").as_double();
    config.stationary_speed_threshold_mps = get_parameter(
      "stationary_speed_threshold_mps").as_double();
    config.wheel_update_beta = get_parameter("wheel_update_beta").as_double();
    config.stationary_hold_s = get_parameter("stationary_hold_s").as_double();
    config.stationary_ax_abs_max_mps2 = get_parameter(
      "stationary_ax_abs_max_mps2").as_double();
    config.stationary_ay_abs_max_mps2 = get_parameter(
      "stationary_ay_abs_max_mps2").as_double();
    config.stationary_yaw_rate_abs_max_radps = get_parameter(
      "stationary_yaw_rate_abs_max_radps").as_double();
    config.turn_enter_yaw_rate_radps = get_parameter(
      "turn_enter_yaw_rate_radps").as_double();
    config.turn_enter_abs_ay_mps2 = get_parameter(
      "turn_enter_abs_ay_mps2").as_double();
    config.turn_exit_yaw_rate_radps = get_parameter(
      "turn_exit_yaw_rate_radps").as_double();
    config.turn_exit_abs_ay_mps2 = get_parameter(
      "turn_exit_abs_ay_mps2").as_double();
    config.turn_exit_hold_s = get_parameter("turn_exit_hold_s").as_double();
    config.turn_wheel_braking_ax_mps2 = get_parameter(
      "turn_wheel_braking_ax_mps2").as_double();
    config.integrate_lateral_acceleration_in_turn = get_parameter(
      "integrate_lateral_acceleration_in_turn").as_bool();
    config.imu_x_offset_m = get_parameter("imu_x_m").as_double();
    config.max_imu_ax_abs_mps2 = get_parameter("max_imu_ax_abs_mps2").as_double();
    return config;
  }

  void reset_diagnostic_epoch()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    packet_assembler_.reset();
    last_processed_stamp_ns_ = 0;
    yaw_initialized_ = false;
    observer_.reset();
    RCLCPP_INFO(get_logger(), "Odometry observer reset to diagnostic origin");
  }

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
    imu.header.stamp = now();
    imu.header.frame_id = base_frame_;
    imu.child_frame_id = imu_frame_;
    imu.transform.translation.x = imu_x_m_;
    imu.transform.translation.y = imu_y_m_;
    imu.transform.translation.z = imu_z_m_;
    imu.transform.rotation.w = 1.0;

    static_tf_broadcaster_->sendTransform({lidar, imu});
  }

  void imu_callback(const sensor_msgs::msg::Imu & msg)
  {
    const double yaw = yaw_from_quaternion(msg.orientation);
    if (!std::isfinite(yaw) || !std::isfinite(msg.linear_acceleration.x) ||
      !std::isfinite(msg.linear_acceleration.y) ||
      !std::isfinite(msg.angular_velocity.z))
    {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    packet_assembler_.add_imu_sample(
      stamp_ns(msg.header.stamp), msg.linear_acceleration.x,
      msg.linear_acceleration.y, msg.angular_velocity.z, yaw);
  }

  void encoder_callback(const sensor_msgs::msg::JointState & msg, bool left)
  {
    if (msg.position.empty() || !std::isfinite(msg.position.front())) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    packet_assembler_.add_encoder_sample(
      stamp_ns(msg.header.stamp), msg.position.front(), left);
  }

  double continuous_yaw(const f1tenth_localization::SensorPacket & packet)
  {
    const double raw_yaw = packet.imu_yaw_rad;
    if (!yaw_initialized_) {
      yaw_initialized_ = true;
      yaw_reference_rad_ = raw_yaw;
      previous_raw_relative_yaw_rad_ = 0.0;
      continuous_yaw_rad_ = 0.0;
      previous_yaw_stamp_s_ = static_cast<double>(packet.stamp_ns) * 1.0e-9;
      previous_yaw_rate_radps_ = packet.yaw_rate_radps;
      return continuous_yaw_rad_;
    }

    const double stamp_s = static_cast<double>(packet.stamp_ns) * 1.0e-9;
    const double dt = stamp_s - previous_yaw_stamp_s_;
    const double raw_relative = wrap_angle(raw_yaw - yaw_reference_rad_);
    const double raw_delta = wrap_angle(raw_relative - previous_raw_relative_yaw_rad_);
    if (dt > 0.0 && dt <= 0.5) {
      continuous_yaw_rad_ = wrap_angle(
        continuous_yaw_rad_ + packet.yaw_rate_radps * dt);
      if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
        continuous_yaw_rad_ = wrap_angle(
          continuous_yaw_rad_ + imu_orientation_correction_gain_ *
          wrap_angle(raw_relative - continuous_yaw_rad_));
      } else {
        yaw_reference_rad_ = wrap_angle(raw_yaw - continuous_yaw_rad_);
        previous_raw_relative_yaw_rad_ = continuous_yaw_rad_;
      }
    }
    if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
      previous_raw_relative_yaw_rad_ = raw_relative;
    }
    previous_yaw_stamp_s_ = stamp_s;
    previous_yaw_rate_radps_ = packet.yaw_rate_radps;
    return continuous_yaw_rad_;
  }

  void process_packet(const f1tenth_localization::SensorPacket & packet)
  {
    if (packet.stamp_ns <= last_processed_stamp_ns_) {
      return;
    }
    last_processed_stamp_ns_ = packet.stamp_ns;

    f1tenth_localization::OdometryObservation observation;
    observation.stamp_s = static_cast<double>(packet.stamp_ns) * 1.0e-9;
    observation.left_angle_rad = packet.left_angle_rad;
    observation.right_angle_rad = packet.right_angle_rad;
    observation.ax_mps2 = packet.ax_mps2;
    observation.ay_mps2 = packet.ay_mps2;
    observation.yaw_rate_radps = packet.yaw_rate_radps;
    observation.yaw_rad = continuous_yaw(packet);

    const auto estimate = observer_.update(observation);
    publish_estimate(estimate, packet.stamp_ns);
  }

  void publish_estimate(
    const f1tenth_localization::OdometryEstimate & estimate,
    int64_t source_stamp_ns)
  {
    const rclcpp::Time stamp(source_stamp_ns, RCL_ROS_TIME);
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = estimate.x_m;
    odom.pose.pose.position.y = estimate.y_m;
    odom.pose.pose.orientation = quaternion_from_yaw(estimate.yaw_rad);
    odom.twist.twist.linear.x = estimate.body_u_mps;
    odom.twist.twist.linear.y = estimate.body_v_mps;
    odom.twist.twist.angular.z = estimate.yaw_rate_radps;

    // A rejected IMU outlier does not move the causal pose. Keep its pose
    // covariance at the normal floor so one discarded sample cannot poison
    // the EKF covariance for the remainder of a run. An isolated timing gap
    // gets a bounded inflation, not the old 0.25 m^2 stop-gate value: the EKF
    // uses this covariance as a floor for the rest of the epoch.
    const double pose_variance = estimate.timing_degraded &&
      !estimate.sensor_outlier ?
      timing_degraded_pose_xy_variance_ : pose_xy_variance_;
    odom.pose.covariance[0] = pose_variance;
    odom.pose.covariance[7] = pose_variance;
    odom.pose.covariance[35] = pose_yaw_variance_;
    odom.twist.covariance[0] = twist_linear_variance_;
    odom.twist.covariance[7] = twist_linear_variance_;
    odom.twist.covariance[35] = twist_yaw_variance_;
    odom_pub_->publish(odom);

    std_msgs::msg::Float64MultiArray diagnostics;
    diagnostics.layout.dim.resize(1);
    diagnostics.layout.dim[0].label = "deterministic_odometry_v4";
    diagnostics.layout.dim[0].size = 27;
    diagnostics.layout.dim[0].stride = 27;
    diagnostics.data = {
      4.0,
      estimate.stamp_s,
      estimate.dt_s,
      estimate.wheel_raw_mps,
      estimate.wheel_mapped_mps,
      estimate.speed_pred_mps,
      estimate.speed_mps,
      estimate.body_u_mps,
      estimate.body_v_mps,
      estimate.ax_mps2,
      estimate.ay_mps2,
      estimate.yaw_rate_radps,
      estimate.wheel_update_used ? 1.0 : 0.0,
      estimate.turn_mode ? 1.0 : 0.0,
      estimate.reset_epoch ? 1.0 : 0.0,
      estimate.timing_degraded ? 1.0 : 0.0,
      static_cast<double>(packet_assembler_.packet_drop_count()),
      static_cast<double>(packet_assembler_.packet_coherence_fault_count()),
      estimate.x_m,
      estimate.y_m,
      estimate.yaw_rad,
      estimate.sensor_outlier ? 1.0 : 0.0,
      estimate.left_angle_rad,
      estimate.right_angle_rad,
      estimate.imu_yaw_rad,
      estimate.wheel_packet_mps,
      estimate.wheel_burst_rejected ? 1.0 : 0.0};
    diagnostics_pub_->publish(diagnostics);

    geometry_msgs::msg::TransformStamped transform;
    transform.header = odom.header;
    transform.child_frame_id = base_frame_;
    transform.transform.translation.x = estimate.x_m;
    transform.transform.translation.y = estimate.y_m;
    transform.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(transform);
  }

  f1tenth_localization::OdometryObserverConfig observer_config_;
  f1tenth_localization::OdometryObserver observer_;
  f1tenth_localization::SensorPacketAssembler packet_assembler_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_sub_;

  std::mutex mutex_;
  std::size_t max_pending_packets_{8};
  int64_t last_processed_stamp_ns_{0};

  bool yaw_initialized_{false};
  double yaw_reference_rad_{0.0};
  double previous_raw_relative_yaw_rad_{0.0};
  double continuous_yaw_rad_{0.0};
  double previous_yaw_stamp_s_{0.0};
  double previous_yaw_rate_radps_{0.0};
  double imu_orientation_correction_gain_{1.0};
  double max_imu_orientation_step_rad_{0.30};

  std::string odom_frame_;
  std::string base_frame_;
  std::string lidar_frame_;
  std::string imu_frame_;
  double lidar_x_m_{0.2733};
  double lidar_y_m_{0.0};
  double lidar_z_m_{0.096};
  double imu_x_m_{0.08};
  double imu_y_m_{0.0};
  double imu_z_m_{0.055};
  double pose_xy_variance_{0.01};
  double timing_degraded_pose_xy_variance_{0.01};
  double pose_yaw_variance_{0.0001};
  double twist_linear_variance_{0.04};
  double twist_yaw_variance_{0.04};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorOdometryNode>());
  rclcpp::shutdown();
  return 0;
}

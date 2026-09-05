#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "f1tenth_localization/odom_estimator_v2_features.hpp"
#include "f1tenth_localization/odom_estimator_v2_model.hpp"
#include "f1tenth_localization/sensor_fusion_model.hpp"
#include "f1tenth_localization/longitudinal_observer.hpp"

namespace {

double wrap_angle(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
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

}  // namespace

using f1tenth_localization::SensorMotionRegime;

class SensorOdometryNode final : public rclcpp::Node
{
public:
  SensorOdometryNode()
  : Node("sensor_odometry"),
    tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
    static_tf_broadcaster_(
      std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this))
  {
    declare_parameter("left_encoder_topic", "/autodrive/roboracer_1/left_encoder");
    declare_parameter("right_encoder_topic", "/autodrive/roboracer_1/right_encoder");
    declare_parameter("imu_topic", "/autodrive/roboracer_1/imu");
    declare_parameter("odom_topic", "/odom");
    declare_parameter("diagnostics_topic", "/odom/diagnostics");
    // Simulator epoch resets are diagnostics-only. Production odometry must
    // remain continuous and leaves this disabled.
    declare_parameter("reset_enabled", false);
    declare_parameter("reset_topic", "/autodrive/reset_command");

    declare_parameter("odom_frame", "odom");
    declare_parameter("base_frame", "base_link");
    declare_parameter("lidar_frame", "lidar");
    declare_parameter("imu_frame", "imu");

    // Fixed documented RoboRacer geometry; this is not runtime-calibrated.
    declare_parameter("wheel_radius_m", 0.0590);
    declare_parameter("encoder_angle_scale", 1.0);
    // Keep the broad distance guard for long native telemetry gaps. The
    // source-time guard below is the important one for the measured ~40 Hz
    // stream: a bridge burst can carry a full wheel increment with a 1-2 ms
    // timestamp interval, which is an impossible wheel speed and must not
    // enter odom.
    declare_parameter("max_encoder_step_m", 4.0);
    declare_parameter("max_encoder_step_speed_mps", 35.0);
    declare_parameter("max_encoder_pair_skew_s", 0.05);
    declare_parameter("lidar_x_m", 0.2733);
    declare_parameter("lidar_y_m", 0.0);
    declare_parameter("lidar_z_m", 0.096);
    declare_parameter("imu_x_m", 0.08);
    declare_parameter("imu_y_m", 0.0);
    declare_parameter("imu_z_m", 0.055);
    // The official orientation is accurate in the simulation.  Use it
    // directly when its step is plausible; gyro integration remains the
    // fallback for an isolated discontinuity.
    declare_parameter("imu_orientation_correction_gain", 1.0);
    declare_parameter("max_imu_orientation_step_rad", 0.30);
    declare_parameter("max_imu_dt_s", 0.5);
    declare_parameter("imu_pair_extrapolation_max_s", 0.10);

    // Longitudinal observer: the documented encoder geometry remains the
    // mechanical conversion, followed by the measured simulator slip map.
    // The map is active from standstill because the clean open-ground sweep
    // shows measurable slip before the old high-speed-only threshold.
    declare_parameter("imu_acceleration_filter_alpha", 0.90);
    // The simulator occasionally emits isolated longitudinal IMU spikes at
    // the native telemetry rate. Keep the raw value for the fitted model
    // feature, but use a short causal median for integration and regime
    // selection so one packet cannot create a false speed/acceleration mode.
    declare_parameter("imu_acceleration_median_window", 3);
    declare_parameter("wheel_slip_threshold_mps", 0.75);
    declare_parameter("wheel_slip_ratio", 0.20);
    // Sx is ill-conditioned near standstill.  Do not classify launch
    // quantisation as tire slip until the IMU speed makes the documented
    // slip ratio observable.
    declare_parameter("slip_observation_min_speed_mps", 0.75);
    declare_parameter("wheel_slip_activation_speed_mps", 0.0);
    declare_parameter("encoder_reset_threshold_rad", 0.50);
    // Innovation gate tuned from the clean truth/IMU response: normal body
    // acceleration is accepted, while a delayed wheel burst is not allowed
    // to jump the body-speed estimate to the wheel-spin speed.
    declare_parameter("max_observer_accel_mps2", 8.0);
    declare_parameter("observer_speed_tolerance_mps", 1.0);
    declare_parameter("wheel_observer_correction_gain", 0.10);
    declare_parameter("observer_acceleration_noise_mps2", 1.50);
    declare_parameter("observer_bias_random_walk_mps3", 0.08);
    declare_parameter("observer_initial_speed_variance_m2ps2", 1.00);
    declare_parameter("observer_initial_bias_variance_m4ps4", 0.25);
    declare_parameter("observer_innovation_gate_mps", 1.50);
    declare_parameter("observer_encoder_measurement_variance_m2ps2", 0.09);
    declare_parameter(
      "observer_accelerating_encoder_measurement_variance_m2ps2", 0.49);
    declare_parameter(
      "observer_steady_encoder_measurement_variance_m2ps2", 0.04);
    declare_parameter(
      "observer_decelerating_encoder_measurement_variance_m2ps2", 0.25);
    declare_parameter("sensor_fusion_model_enabled", true);
    // v2 is the promoted causal residual estimator from the attached
    // holdout-validated handoff. The older generated regime forest remains
    // available only as a source-compatible fallback when this is disabled.
    declare_parameter("sensor_fusion_v2_enabled", true);
    declare_parameter("sensor_fusion_v2_braking_blend_offset_mps2", 0.35);
    declare_parameter("sensor_fusion_v2_braking_blend_range_mps2", 0.50);
    // In the steady regime the bounded wheel/IMU baseline is already an
    // absolute observation. The learned steady branch is retained for
    // diagnostics but is disabled by default because it can turn a valid
    // low-speed wheel update into a stale launch-speed bias.
    declare_parameter("sensor_fusion_steady_model_enabled", false);
    declare_parameter("regime_acceleration_threshold_mps2", 0.50);
    declare_parameter("regime_enter_acceleration_mps2", 0.65);
    declare_parameter("regime_exit_acceleration_mps2", 0.25);
    declare_parameter("regime_min_dwell_s", 0.075);
    declare_parameter("sensor_fusion_window_low_speed_duration_s", 1.00);
    declare_parameter("sensor_fusion_window_mid_speed_duration_s", 0.60);
    declare_parameter("sensor_fusion_window_high_speed_duration_s", 0.30);
    declare_parameter("sensor_fusion_window_low_speed_threshold_mps", 5.0);
    declare_parameter("sensor_fusion_window_high_speed_threshold_mps", 15.0);
    declare_parameter("sensor_fusion_window_stability_mps", 0.35);
    // A throttle downshift can make the driven wheel speed change before the
    // body has decelerated. Hold the IMU estimate through that causal
    // transient instead of accepting the new wheel speed as body speed.
    declare_parameter("sensor_fusion_wheel_downshift_min_drop_mps", 0.35);
    declare_parameter("sensor_fusion_wheel_downshift_hold_s", 0.50);
    // Do not reuse a cumulative encoder window across a braking interval.
    // The driven wheel can freeze while the body keeps moving, which would
    // otherwise dilute the first steady low-speed samples after braking.
    declare_parameter("sensor_fusion_window_recovery_s", 0.35);
    declare_parameter("sensor_fusion_recovery_stability_mps", 0.50);
    declare_parameter("sensor_fusion_recovery_encoder_blend", 0.75);
    declare_parameter("sensor_fusion_recovery_encoder_max_correction_mps", 0.35);
    declare_parameter("sensor_fusion_model_blend", 0.50);
    declare_parameter("sensor_fusion_transient_model_blend", 0.50);
    declare_parameter("sensor_fusion_transient_model_max_correction_mps", 0.35);
    declare_parameter("sensor_fusion_transient_low_speed_threshold_mps", 0.75);
    declare_parameter("sensor_fusion_transient_mid_speed_threshold_mps", 1.50);
    declare_parameter("sensor_fusion_transient_low_speed_max_correction_mps", 0.10);
    declare_parameter("sensor_fusion_transient_mid_speed_max_correction_mps", 0.20);
    // At low speed the timestamped encoder window is better conditioned than
    // an IMU integral with an unknown bias.  Only promote it when it is
    // stable and close to the IMU prediction; this cannot turn a large wheel
    // spin burst into body speed.
    declare_parameter("low_speed_encoder_priority_mps", 5.0);
    declare_parameter("low_speed_encoder_max_imu_gap_mps", 0.75);
    declare_parameter("low_speed_encoder_stability_mps", 0.35);
    declare_parameter("low_speed_encoder_stability_samples", 8);
    declare_parameter("regime_model_max_spread_mps", 0.75);
    declare_parameter("regime_model_max_baseline_delta_mps", 0.75);
    // A quiet, consistent wheel/map measurement below the IMU observer is
    // evidence of accumulated positive IMU-speed bias rather than driven
    // wheel overspeed. Allow that sensor-only branch to re-anchor the
    // observer without using throttle, ground truth, or AMCL.
    declare_parameter("steady_encoder_reanchor_enabled", true);
    // Keep the IMU state independent until a live bias-correction fit is
    // accepted; the YAML candidate currently leaves this at zero.
    declare_parameter("imu_speed_correction_gain", 0.0);
    declare_parameter("stationary_speed_threshold_mps", 0.15);
    // AutoDRIVE documents the longitudinal WheelCollider curve using these
    // slip/force knots.  Runtime estimation uses the extremum/asymptote slip
    // boundaries to decide when wheel speed is no longer a valid body-speed
    // observation; the force values are retained for validation and model
    // identification, not treated as a hidden acceleration measurement.
    declare_parameter("forward_extremum_slip", 0.15);
    declare_parameter("forward_extremum_value", 0.72);
    declare_parameter("forward_asymptote_slip", 0.25);
    declare_parameter("forward_asymptote_value", 0.464);
    // When the driven-wheel encoder freezes during simulator coasting, use
    // the allowed IMU integration until a low-speed/low-acceleration stop is
    // confirmed.  This avoids cutting off the coast distance while preventing
    // residual IMU bias from producing indefinite motion at rest.
    // A stop confirmation must not classify a low-speed launch as stopped.
    // Keep this close to the stationary threshold; the old 2 m/s bound could
    // zero a moving observer while the first encoder window was still empty.
    declare_parameter("imu_stop_speed_threshold_mps", 0.20);
    declare_parameter("imu_stationary_acceleration_threshold_mps2", 0.30);
    declare_parameter("zero_encoder_stop_confirm_sec", 0.80);
    // Offline identification found a bounded braking deceleration trend when
    // the driven encoder is frozen. This is a sensor-only fallback for the
    // unobservable case where the IMU goes quiet before its integrated speed
    // has decayed; it predicts remaining slip distance and never teleports
    // pose to zero.
    declare_parameter("frozen_encoder_model_enabled", true);
    declare_parameter("frozen_encoder_model_delay_sec", 0.15);
    declare_parameter("frozen_encoder_decel_intercept_mps2", 5.5);
    declare_parameter("frozen_encoder_decel_speed_gain_per_s", 0.27);
    declare_parameter("frozen_encoder_decel_max_mps2", 12.0);
    // During an active brake the IMU remains the primary measurement, but the
    // open-world fit shows a small, repeatable under-estimation of the true
    // deceleration. Apply only a positive, bounded correction toward the
    // frozen-encoder prior; never replace a stronger measured IMU brake.
    declare_parameter("frozen_encoder_braking_model_blend", 0.75);
    declare_parameter("slip_pose_xy_variance", 0.10);
    declare_parameter("encoder_reset_covariance_duration_s", 1.0);
    declare_parameter("encoder_reset_pose_xy_variance", 0.25);
    declare_parameter("encoder_reset_twist_linear_variance", 0.50);
    declare_parameter(
      "wheel_speed_map_wheel_mps",
      std::vector<double>{0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
        16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0});
    declare_parameter(
      "wheel_speed_map_body_mps",
      std::vector<double>{0.0, 1.988601, 3.967925, 5.888532, 8.549802,
        9.662430, 11.536114, 13.473328, 15.270020, 16.855770,
        18.570086, 19.279928, 19.987528, 20.287600, 20.287600,
        20.287600});

    declare_parameter("pose_xy_variance", 0.01);
    declare_parameter("pose_yaw_variance", 0.01);
    declare_parameter("velocity_filter_alpha", 1.0);
    declare_parameter("velocity_median_window", 3);
    declare_parameter("velocity_filter_decel_alpha", 1.0);
    // Outlier guard only; do not make this a hidden low-speed ceiling.
    declare_parameter("max_velocity_accel_mps2", 40.0);
    declare_parameter("twist_linear_variance", 0.04);
    declare_parameter("twist_yaw_variance", 0.04);

    wheel_radius_ = get_parameter("wheel_radius_m").as_double();
    encoder_scale_ = get_parameter("encoder_angle_scale").as_double();
    max_encoder_step_m_ = get_parameter("max_encoder_step_m").as_double();
    max_encoder_step_speed_mps_ = std::max(
      1.0, get_parameter("max_encoder_step_speed_mps").as_double());
    max_encoder_pair_skew_s_ = std::max(
      0.0, get_parameter("max_encoder_pair_skew_s").as_double());
    pose_xy_var_ = get_parameter("pose_xy_variance").as_double();
    pose_yaw_var_ = get_parameter("pose_yaw_variance").as_double();
    velocity_filter_alpha_ = std::clamp(
      get_parameter("velocity_filter_alpha").as_double(), 0.01, 1.0);
    velocity_filter_decel_alpha_ = std::clamp(
      get_parameter("velocity_filter_decel_alpha").as_double(), 0.01, 1.0);
    const auto median_window = get_parameter("velocity_median_window").as_int();
    velocity_median_window_ = static_cast<size_t>(std::max<int64_t>(
      1, std::min<int64_t>(9, median_window)));
    max_velocity_accel_mps2_ = std::max(
      0.1, get_parameter("max_velocity_accel_mps2").as_double());
    twist_linear_var_ = get_parameter("twist_linear_variance").as_double();
    twist_yaw_var_ = get_parameter("twist_yaw_variance").as_double();
    imu_orientation_correction_gain_ = std::clamp(
      get_parameter("imu_orientation_correction_gain").as_double(), 0.0, 1.0);
    max_imu_orientation_step_rad_ = std::max(
      0.05, get_parameter("max_imu_orientation_step_rad").as_double());
    max_imu_dt_s_ = std::max(0.05, get_parameter("max_imu_dt_s").as_double());
    imu_pair_extrapolation_max_s_ = std::max(
      0.0, get_parameter("imu_pair_extrapolation_max_s").as_double());
    imu_acceleration_filter_alpha_ = std::clamp(
      get_parameter("imu_acceleration_filter_alpha").as_double(), 0.01, 1.0);
    const auto imu_median_window = get_parameter(
      "imu_acceleration_median_window").as_int();
    imu_acceleration_median_window_ = static_cast<size_t>(std::max<int64_t>(
      1, std::min<int64_t>(9, imu_median_window)));
    wheel_slip_threshold_mps_ = std::max(
      0.0, get_parameter("wheel_slip_threshold_mps").as_double());
    wheel_slip_ratio_ = std::max(
      0.0, get_parameter("wheel_slip_ratio").as_double());
    stationary_speed_threshold_mps_ = std::max(
      0.0, get_parameter("stationary_speed_threshold_mps").as_double());
    slip_observation_min_speed_mps_ = std::max(
      stationary_speed_threshold_mps_, get_parameter(
        "slip_observation_min_speed_mps").as_double());
    wheel_slip_activation_speed_mps_ = std::max(
      0.0, get_parameter("wheel_slip_activation_speed_mps").as_double());
    encoder_reset_threshold_rad_ = std::max(
      0.05, get_parameter("encoder_reset_threshold_rad").as_double());
    max_observer_accel_mps2_ = std::max(
      0.1, get_parameter("max_observer_accel_mps2").as_double());
    observer_speed_tolerance_mps_ = std::max(
      0.0, get_parameter("observer_speed_tolerance_mps").as_double());
    wheel_observer_correction_gain_ = std::clamp(
      get_parameter("wheel_observer_correction_gain").as_double(), 0.0, 1.0);
    observer_acceleration_noise_mps2_ = std::max(
      0.01, get_parameter("observer_acceleration_noise_mps2").as_double());
    observer_bias_random_walk_mps3_ = std::max(
      0.0001, get_parameter("observer_bias_random_walk_mps3").as_double());
    observer_initial_speed_variance_m2ps2_ = std::max(
      1.0e-6, get_parameter("observer_initial_speed_variance_m2ps2").as_double());
    observer_initial_bias_variance_m4ps4_ = std::max(
      1.0e-6, get_parameter("observer_initial_bias_variance_m4ps4").as_double());
    observer_innovation_gate_mps_ = std::max(
      0.0, get_parameter("observer_innovation_gate_mps").as_double());
    observer_encoder_measurement_variance_m2ps2_ = std::max(
      1.0e-6, get_parameter(
        "observer_encoder_measurement_variance_m2ps2").as_double());
    observer_accelerating_measurement_variance_m2ps2_ = std::max(
      1.0e-6, get_parameter(
        "observer_accelerating_encoder_measurement_variance_m2ps2").as_double());
    observer_steady_measurement_variance_m2ps2_ = std::max(
      1.0e-6, get_parameter(
        "observer_steady_encoder_measurement_variance_m2ps2").as_double());
    observer_decelerating_measurement_variance_m2ps2_ = std::max(
      1.0e-6, get_parameter(
        "observer_decelerating_encoder_measurement_variance_m2ps2").as_double());
    sensor_fusion_model_enabled_ = get_parameter(
      "sensor_fusion_model_enabled").as_bool();
    sensor_fusion_v2_enabled_ = get_parameter(
      "sensor_fusion_v2_enabled").as_bool();
    sensor_fusion_v2_braking_blend_offset_mps2_ = std::max(
      0.0, get_parameter("sensor_fusion_v2_braking_blend_offset_mps2").as_double());
    sensor_fusion_v2_braking_blend_range_mps2_ = std::max(
      1.0e-6, get_parameter("sensor_fusion_v2_braking_blend_range_mps2").as_double());
    sensor_fusion_steady_model_enabled_ = get_parameter(
      "sensor_fusion_steady_model_enabled").as_bool();
    regime_acceleration_threshold_mps2_ = std::max(
      0.05, get_parameter("regime_acceleration_threshold_mps2").as_double());
    regime_enter_acceleration_mps2_ = std::max(
      regime_acceleration_threshold_mps2_, get_parameter(
        "regime_enter_acceleration_mps2").as_double());
    regime_exit_acceleration_mps2_ = std::clamp(
      get_parameter("regime_exit_acceleration_mps2").as_double(),
      0.01, regime_enter_acceleration_mps2_);
    regime_min_dwell_s_ = std::max(
      0.0, get_parameter("regime_min_dwell_s").as_double());
    sensor_fusion_window_low_speed_duration_s_ = std::max(
      0.10, get_parameter("sensor_fusion_window_low_speed_duration_s").as_double());
    sensor_fusion_window_mid_speed_duration_s_ = std::max(
      0.10, get_parameter("sensor_fusion_window_mid_speed_duration_s").as_double());
    sensor_fusion_window_high_speed_duration_s_ = std::max(
      0.10, get_parameter("sensor_fusion_window_high_speed_duration_s").as_double());
    sensor_fusion_window_low_speed_threshold_mps_ = std::max(
      0.10, get_parameter("sensor_fusion_window_low_speed_threshold_mps").as_double());
    sensor_fusion_window_high_speed_threshold_mps_ = std::max(
      sensor_fusion_window_low_speed_threshold_mps_, get_parameter(
        "sensor_fusion_window_high_speed_threshold_mps").as_double());
    sensor_fusion_window_stability_mps_ = std::max(
      0.0, get_parameter("sensor_fusion_window_stability_mps").as_double());
    sensor_fusion_wheel_downshift_min_drop_mps_ = std::max(
      0.05, get_parameter(
        "sensor_fusion_wheel_downshift_min_drop_mps").as_double());
    sensor_fusion_wheel_downshift_hold_s_ = std::max(
      0.0, get_parameter("sensor_fusion_wheel_downshift_hold_s").as_double());
    sensor_fusion_window_recovery_s_ = std::max(
      0.0, get_parameter("sensor_fusion_window_recovery_s").as_double());
    sensor_fusion_recovery_stability_mps_ = std::max(
      0.0, get_parameter("sensor_fusion_recovery_stability_mps").as_double());
    sensor_fusion_recovery_encoder_blend_ = std::clamp(
      get_parameter("sensor_fusion_recovery_encoder_blend").as_double(), 0.0, 1.0);
    sensor_fusion_recovery_encoder_max_correction_mps_ = std::max(
      0.0, get_parameter(
        "sensor_fusion_recovery_encoder_max_correction_mps").as_double());
    sensor_fusion_model_blend_ = std::clamp(
      get_parameter("sensor_fusion_model_blend").as_double(), 0.0, 1.0);
    sensor_fusion_transient_model_blend_ = std::clamp(
      get_parameter("sensor_fusion_transient_model_blend").as_double(), 0.0, 1.0);
    sensor_fusion_transient_model_max_correction_mps_ = std::max(
      0.0, get_parameter("sensor_fusion_transient_model_max_correction_mps").as_double());
    sensor_fusion_transient_low_speed_threshold_mps_ = std::max(
      stationary_speed_threshold_mps_, get_parameter(
        "sensor_fusion_transient_low_speed_threshold_mps").as_double());
    sensor_fusion_transient_mid_speed_threshold_mps_ = std::max(
      sensor_fusion_transient_low_speed_threshold_mps_, get_parameter(
        "sensor_fusion_transient_mid_speed_threshold_mps").as_double());
    sensor_fusion_transient_low_speed_max_correction_mps_ = std::max(
      0.0, get_parameter(
        "sensor_fusion_transient_low_speed_max_correction_mps").as_double());
    sensor_fusion_transient_mid_speed_max_correction_mps_ = std::max(
      sensor_fusion_transient_low_speed_max_correction_mps_, get_parameter(
        "sensor_fusion_transient_mid_speed_max_correction_mps").as_double());
    low_speed_encoder_priority_mps_ = std::max(
      stationary_speed_threshold_mps_, get_parameter(
        "low_speed_encoder_priority_mps").as_double());
    low_speed_encoder_max_imu_gap_mps_ = std::max(
      0.0, get_parameter("low_speed_encoder_max_imu_gap_mps").as_double());
    low_speed_encoder_stability_mps_ = std::max(
      0.0, get_parameter("low_speed_encoder_stability_mps").as_double());
    low_speed_encoder_stability_samples_ = std::max<std::size_t>(
      3, static_cast<std::size_t>(get_parameter(
        "low_speed_encoder_stability_samples").as_int()));
    regime_model_max_spread_mps_ = std::max(
      0.0, get_parameter("regime_model_max_spread_mps").as_double());
    regime_model_max_baseline_delta_mps_ = std::max(
      0.0, get_parameter("regime_model_max_baseline_delta_mps").as_double());
    steady_encoder_reanchor_enabled_ = get_parameter(
      "steady_encoder_reanchor_enabled").as_bool();
    imu_speed_correction_gain_ = std::clamp(
      get_parameter("imu_speed_correction_gain").as_double(), 0.0, 1.0);
    forward_extremum_slip_ = get_parameter("forward_extremum_slip").as_double();
    forward_extremum_value_ = get_parameter("forward_extremum_value").as_double();
    forward_asymptote_slip_ = get_parameter("forward_asymptote_slip").as_double();
    forward_asymptote_value_ = get_parameter("forward_asymptote_value").as_double();
    if (!std::isfinite(forward_extremum_slip_) ||
      !std::isfinite(forward_extremum_value_) ||
      !std::isfinite(forward_asymptote_slip_) ||
      !std::isfinite(forward_asymptote_value_) ||
      forward_extremum_slip_ <= 0.0 ||
      forward_asymptote_slip_ <= forward_extremum_slip_ ||
      forward_extremum_value_ <= 0.0 ||
      forward_asymptote_value_ <= 0.0 ||
      forward_asymptote_value_ > forward_extremum_value_) {
      throw std::runtime_error("invalid documented longitudinal friction curve");
    }
    imu_stop_speed_threshold_mps_ = std::max(
      stationary_speed_threshold_mps_, get_parameter(
        "imu_stop_speed_threshold_mps").as_double());
    imu_stationary_acceleration_threshold_mps2_ = std::max(
      0.0, get_parameter("imu_stationary_acceleration_threshold_mps2").as_double());
    zero_encoder_stop_confirm_sec_ = std::max(
      0.0, get_parameter("zero_encoder_stop_confirm_sec").as_double());
    frozen_encoder_model_enabled_ = get_parameter(
      "frozen_encoder_model_enabled").as_bool();
    frozen_encoder_model_delay_sec_ = std::max(
      0.0, get_parameter("frozen_encoder_model_delay_sec").as_double());
    frozen_encoder_decel_intercept_mps2_ = std::max(
      0.0, get_parameter("frozen_encoder_decel_intercept_mps2").as_double());
    frozen_encoder_decel_speed_gain_per_s_ = std::max(
      0.0, get_parameter("frozen_encoder_decel_speed_gain_per_s").as_double());
    frozen_encoder_decel_max_mps2_ = std::max(
      0.1, get_parameter("frozen_encoder_decel_max_mps2").as_double());
    frozen_encoder_braking_model_blend_ = std::clamp(
      get_parameter("frozen_encoder_braking_model_blend").as_double(), 0.0, 1.0);
    if (frozen_encoder_decel_intercept_mps2_ > frozen_encoder_decel_max_mps2_) {
      throw std::runtime_error("frozen encoder deceleration intercept exceeds maximum");
    }
    slip_pose_xy_var_ = std::max(
      pose_xy_var_, get_parameter("slip_pose_xy_variance").as_double());
    encoder_reset_covariance_duration_s_ = std::max(
      0.0, get_parameter("encoder_reset_covariance_duration_s").as_double());
    encoder_reset_pose_xy_var_ = std::max(
      pose_xy_var_, get_parameter("encoder_reset_pose_xy_variance").as_double());
    encoder_reset_twist_linear_var_ = std::max(
      twist_linear_var_, get_parameter("encoder_reset_twist_linear_variance").as_double());
    wheel_speed_map_wheel_mps_ = get_parameter(
      "wheel_speed_map_wheel_mps").as_double_array();
    wheel_speed_map_body_mps_ = get_parameter(
      "wheel_speed_map_body_mps").as_double_array();
    if (wheel_speed_map_wheel_mps_.size() < 2 ||
        wheel_speed_map_wheel_mps_.size() != wheel_speed_map_body_mps_.size()) {
      throw std::runtime_error("wheel-speed slip maps must have equal length >= 2");
    }
    for (size_t index = 0; index < wheel_speed_map_wheel_mps_.size(); ++index) {
      if (!std::isfinite(wheel_speed_map_wheel_mps_[index]) ||
          !std::isfinite(wheel_speed_map_body_mps_[index]) ||
          wheel_speed_map_wheel_mps_[index] < 0.0 ||
          wheel_speed_map_body_mps_[index] < 0.0 ||
          (index > 0 && wheel_speed_map_wheel_mps_[index] <=
            wheel_speed_map_wheel_mps_[index - 1]) ||
          (index > 0 && wheel_speed_map_body_mps_[index] <
            wheel_speed_map_body_mps_[index - 1])) {
        throw std::runtime_error("wheel-speed slip maps must be finite and monotonic");
      }
    }
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    longitudinal_observer_.configure(
      observer_acceleration_noise_mps2_, observer_bias_random_walk_mps3_,
      observer_initial_speed_variance_m2ps2_, observer_initial_bias_variance_m4ps4_,
      30.0);

    if (wheel_radius_ <= 0.0 || encoder_scale_ <= 0.0 ||
        max_encoder_step_m_ <= 0.0 || max_encoder_step_speed_mps_ <= 0.0) {
      throw std::runtime_error("invalid sensor odometry parameters");
    }

    // Derived odometry is consumed by the recorder, EKF, and timing gate.
    // Keep a bounded reliable backlog so a short Python executor/CSV flush
    // stall cannot drop native derived samples from the dataset.
    const auto derived_qos = rclcpp::QoS(100);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("odom_topic").as_string(), derived_qos);
    diagnostics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      get_parameter("diagnostics_topic").as_string(), derived_qos);

    auto sensor_qos = rclcpp::SensorDataQoS().keep_last(5);
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
      std::bind(&SensorOdometryNode::imu_callback, this, std::placeholders::_1));
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
      "Allowed sensor odometry: rear encoders + IMU -> /odom");
  }

private:

  void reset_diagnostic_epoch()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // The calibration harness verifies that the vehicle has stopped before
    // sending this event. Rebaseline every local state variable so /odom
    // starts at the simulator spawn for the next independent grid point.
    have_left_ = false;
    have_right_ = false;
    left_updated_ = false;
    right_updated_ = false;
    encoder_initialized_ = false;
    imu_initialized_ = false;
    left_angle_ = 0.0;
    right_angle_ = 0.0;
    prev_left_ = 0.0;
    prev_right_ = 0.0;
    left_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    right_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    prev_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    imu_yaw_zero_ = 0.0;
    imu_raw_yaw_ = 0.0;
    last_imu_relative_yaw_ = 0.0;
    imu_integrated_yaw_ = 0.0;
    imu_yaw_rate_ = 0.0;
    odom_yaw_ = 0.0;
    prev_yaw_ = 0.0;
    last_imu_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    have_imu_stamp_ = false;
    speed_mps_ = 0.0;
    recent_raw_speeds_.clear();
    raw_wheel_speed_mps_ = 0.0;
    corrected_wheel_speed_mps_ = 0.0;
    longitudinal_slip_ = 0.0;
    wheel_slip_detected_ = false;
    wheel_observation_confidence_ = 1.0;
    frozen_encoder_model_active_ = false;
    frozen_encoder_model_decel_mps2_ = 0.0;
    sensor_fusion_window_raw_speed_mps_ = 0.0;
    sensor_fusion_window_mapped_speed_mps_ = 0.0;
    sensor_fusion_model_active_ = false;
    sensor_fusion_model_speed_mps_ = 0.0;
    sensor_fusion_model_spread_mps_ = 0.0;
    sensor_fusion_v2_global_speed_mps_ = 0.0;
    sensor_fusion_v2_base_speed_mps_ = 0.0;
    sensor_fusion_v2_braking_speed_mps_ = 0.0;
    sensor_fusion_v2_braking_blend_ = 0.0;
    sensor_fusion_v2_global_residual_ = 0.0;
    sensor_fusion_v2_braking_residual_ = 0.0;
    sensor_fusion_v2_active_ = false;
    sensor_fusion_v2_features_.reset();
    motion_regime_ = SensorMotionRegime::STEADY;
    sensor_fusion_raw_history_.clear();
    sensor_fusion_mapped_history_.clear();
    sensor_fusion_imu_history_.clear();
    sensor_fusion_gap_history_.clear();
    sensor_fusion_encoder_history_.clear();
    sensor_fusion_recovery_window_history_.clear();
    sensor_fusion_braking_window_active_ = false;
    sensor_fusion_window_recovery_elapsed_s_ = 0.0;
    sensor_fusion_wheel_reacquisition_required_ = false;
    steady_mapped_speed_history_.clear();
    zero_encoder_duration_s_ = 0.0;
    last_motion_sign_ = 1.0;
    imu_raw_acceleration_mps2_ = 0.0;
    imu_lateral_acceleration_mps2_ = 0.0;
    x_ = 0.0;
    y_ = 0.0;
    reset_longitudinal_observer();
    encoder_reset_active_ = false;
    RCLCPP_INFO(get_logger(), "Diagnostic odometry epoch reset to spawn origin");
  }

  void publish_static_transforms()
  {
    geometry_msgs::msg::TransformStamped lidar;
    lidar.header.stamp = now();
    lidar.header.frame_id = base_frame_;
    lidar.child_frame_id = get_parameter("lidar_frame").as_string();
    lidar.transform.translation.x = get_parameter("lidar_x_m").as_double();
    lidar.transform.translation.y = get_parameter("lidar_y_m").as_double();
    lidar.transform.translation.z = get_parameter("lidar_z_m").as_double();
    lidar.transform.rotation.w = 1.0;

    geometry_msgs::msg::TransformStamped imu;
    imu.header.stamp = now();
    imu.header.frame_id = base_frame_;
    imu.child_frame_id = get_parameter("imu_frame").as_string();
    imu.transform.translation.x = get_parameter("imu_x_m").as_double();
    imu.transform.translation.y = get_parameter("imu_y_m").as_double();
    imu.transform.translation.z = get_parameter("imu_z_m").as_double();
    imu.transform.rotation.w = 1.0;

    static_tf_broadcaster_->sendTransform({lidar, imu});
  }

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
  {
    const double raw_yaw = yaw_from_quaternion(msg->orientation);
    const double gyro_z = msg->angular_velocity.z;
    const double longitudinal_acceleration = msg->linear_acceleration.x;
    if (!std::isfinite(raw_yaw) || !std::isfinite(gyro_z)) {
      return;
    }

    const rclcpp::Time stamp(msg->header.stamp, get_clock()->get_clock_type());
    std::lock_guard<std::mutex> lock(mutex_);
    imu_raw_yaw_ = raw_yaw;
    if (!imu_initialized_) {
      imu_yaw_zero_ = raw_yaw;
      last_imu_relative_yaw_ = 0.0;
      imu_integrated_yaw_ = 0.0;
      last_imu_stamp_ = stamp;
      have_imu_stamp_ = true;
      odom_yaw_ = 0.0;
      imu_initialized_ = true;
    } else {
      const double raw_relative_yaw = wrap_angle(raw_yaw - imu_yaw_zero_);
      const double raw_delta = wrap_angle(raw_relative_yaw - last_imu_relative_yaw_);
      double dt = 0.0;
      if (have_imu_stamp_) {
        dt = (stamp - last_imu_stamp_).seconds();
      }
      if (dt > 1.0e-4 && dt <= max_imu_dt_s_) {
        const double gyro_delta = gyro_z * dt;
        imu_integrated_yaw_ = wrap_angle(imu_integrated_yaw_ + gyro_delta);

        // AutoDRIVE occasionally emits an orientation sample with a large
        // discontinuity although its gyro stream remains smooth. This also
        // occurs when the simulator teleports the vehicle during a diagnostic
        // reset. Never let that sample, or the next correction, teleport
        // odometry. Rebase the raw orientation while preserving the integrated
        // yaw and let the gyro continue from the continuous state.
        if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
          const double correction = wrap_angle(raw_relative_yaw - imu_integrated_yaw_);
          imu_integrated_yaw_ = wrap_angle(
            imu_integrated_yaw_ + imu_orientation_correction_gain_ * correction);
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Ignoring discontinuous IMU yaw sample (step=%.3f rad); using gyro integration",
            raw_delta);
          imu_yaw_zero_ = wrap_angle(raw_yaw - imu_integrated_yaw_);
          last_imu_relative_yaw_ = imu_integrated_yaw_;
        }
      }
      odom_yaw_ = imu_integrated_yaw_;
      if (std::abs(raw_delta) <= max_imu_orientation_step_rad_) {
        last_imu_relative_yaw_ = raw_relative_yaw;
      }
      last_imu_stamp_ = stamp;
      have_imu_stamp_ = true;
    }
    imu_yaw_rate_ = gyro_z;
    imu_raw_acceleration_mps2_ = std::isfinite(longitudinal_acceleration) ?
      longitudinal_acceleration : 0.0;
    imu_lateral_acceleration_mps2_ = std::isfinite(msg->linear_acceleration.y) ?
      msg->linear_acceleration.y : 0.0;

    update_longitudinal_observer(longitudinal_acceleration, stamp);

    try_integrate_pair();
  }

  void update_longitudinal_observer(
    double acceleration_mps2, const rclcpp::Time & stamp)
  {
    if (!std::isfinite(acceleration_mps2)) {
      return;
    }
    const double acceleration = std::clamp(acceleration_mps2, -25.0, 25.0);
    const double observer_acceleration_input = acceleration;
    const double robust_acceleration = sensor_history_median(
      imu_acceleration_history_, acceleration);
    imu_acceleration_history_.push_back(acceleration);
    while (imu_acceleration_history_.size() > imu_acceleration_median_window_) {
      imu_acceleration_history_.pop_front();
    }
    if (!have_imu_speed_stamp_) {
      imu_acceleration_filtered_mps2_ = observer_acceleration_input;
      imu_acceleration_robust_mps2_ = robust_acceleration;
      imu_observer_acceleration_mps2_ = 0.5 * (
        acceleration + robust_acceleration);
      longitudinal_observer_.reset();
      imu_speed_mps_ = longitudinal_observer_.speed();
      last_imu_speed_stamp_ = stamp;
      have_imu_speed_stamp_ = true;
      imu_speed_ready_ = true;
      return;
    }

    const double dt = (stamp - last_imu_speed_stamp_).seconds();
    if (dt > 1.0e-4 && dt <= max_imu_dt_s_) {
      const double previous_observer_acceleration =
        imu_observer_acceleration_mps2_;
      imu_acceleration_filtered_mps2_ =
        imu_acceleration_filter_alpha_ * observer_acceleration_input +
        (1.0 - imu_acceleration_filter_alpha_) * imu_acceleration_filtered_mps2_;
      imu_acceleration_robust_mps2_ = robust_acceleration;
      // Use the causal median to reject an isolated bridge spike, but do not
      // halve a genuine brake/launch onset while the EMA is already beyond
      // the regime threshold.  The old fixed 50/50 blend delayed the first
      // braking samples by one to two native packets; with a frozen driven
      // encoder that delay became a visible speed error.  Once the filtered
      // signal is inside the regime envelope it remains median-protected.
      const bool filtered_regime_signal =
        std::abs(imu_acceleration_filtered_mps2_) >=
        regime_enter_acceleration_mps2_;
      imu_observer_acceleration_mps2_ = filtered_regime_signal ?
        imu_acceleration_filtered_mps2_ : 0.5 * (
        imu_acceleration_filtered_mps2_ + imu_acceleration_robust_mps2_);
      // Acceleration is a sample at the end of this interval. A trapezoidal
      // step prevents a newly observed brake/release impulse from being
      // applied across the entire preceding interval.
      longitudinal_observer_.predict(0.5 * (
        previous_observer_acceleration + imu_observer_acceleration_mps2_), dt);
      imu_speed_mps_ = longitudinal_observer_.speed();
    }
    last_imu_speed_stamp_ = stamp;
    have_imu_speed_stamp_ = true;
    imu_speed_ready_ = true;
  }

  void encoder_callback(const sensor_msgs::msg::JointState & msg, bool left)
  {
    if (msg.position.empty()) {
      return;
    }
    const double angle = msg.position.front() * encoder_scale_;
    if (!std::isfinite(angle)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (left) {
      left_angle_ = angle;
      left_stamp_ = rclcpp::Time(msg.header.stamp, get_clock()->get_clock_type());
      have_left_ = true;
      left_updated_ = true;
    } else {
      right_angle_ = angle;
      right_stamp_ = rclcpp::Time(msg.header.stamp, get_clock()->get_clock_type());
      have_right_ = true;
      right_updated_ = true;
    }

    // Encoder callbacks can arrive after the IMU callback for the same
    // simulator cycle. Try here as well as from imu_callback so a complete
    // pair is not delayed until the next IMU event. The helper's source-time
    // guard still holds a future encoder pair until its matching IMU exists.
    try_integrate_pair();

  }

  void try_integrate_pair()
  {
    if (!have_left_ || !have_right_ || !left_updated_ || !right_updated_ ||
      !imu_initialized_ || !have_imu_speed_stamp_)
    {
      return;
    }

    const rclcpp::Time pair_stamp = left_stamp_ > right_stamp_ ? left_stamp_ : right_stamp_;
    // The bridge topics do not share a DDS delivery order. An encoder
    // message carrying the next simulator timestamp can reach this node
    // before the IMU callback for that same timestamp. Integrating it here
    // would combine a future encoder displacement with the previous IMU
    // state, then publish an odom sample stamped in the future. Keep the
    // pair pending until the observer has processed an IMU at or beyond
    // the pair timestamp. This is a source-time guard, not a wall-clock
    // rate assumption.
    if (pair_stamp > last_imu_speed_stamp_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Waiting for matching IMU timestamp (encoder=%.6f imu=%.6f)",
        pair_stamp.seconds(), last_imu_speed_stamp_.seconds());
      return;
    }

    const double pair_skew = std::abs((left_stamp_ - right_stamp_).seconds());
    if (pair_skew > max_encoder_pair_skew_s_) {
      // Keep the newer sample and wait for the other wheel from the same
      // bridge cycle. Never combine a delayed wheel message with a newer
      // one, which would create false yaw/velocity spikes.
      if (left_stamp_ < right_stamp_) {
        left_updated_ = false;
      } else {
        right_updated_ = false;
      }
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Waiting for synchronized encoder pair (skew=%.3f s)", pair_skew);
      return;
    }

    integrate_pair();
    left_updated_ = false;
    right_updated_ = false;
  }

  void integrate_pair()
  {
    const rclcpp::Time stamp = left_stamp_ > right_stamp_ ? left_stamp_ : right_stamp_;
    const double pair_skew = std::abs((left_stamp_ - right_stamp_).seconds());
    frozen_encoder_model_active_ = false;
    frozen_encoder_model_decel_mps2_ = 0.0;

    if (!encoder_initialized_) {
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      prev_yaw_ = odom_yaw_;
      speed_mps_ = 0.0;
      reset_longitudinal_observer();
      recent_raw_speeds_.clear();
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      sensor_fusion_encoder_history_.clear();
      sensor_fusion_recovery_window_history_.clear();
      sensor_fusion_v2_features_.reset();
      steady_mapped_speed_history_.clear();
      raw_wheel_speed_mps_ = 0.0;
      corrected_wheel_speed_mps_ = 0.0;
      longitudinal_slip_ = 0.0;
      wheel_observation_confidence_ = 1.0;
      encoder_initialized_ = true;
      publish_odom(stamp, 0.0);
      return;
    }

    const double dt = (stamp - prev_stamp_).seconds();
    if (dt <= 1.0e-4 || dt > 0.5) {
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      prev_yaw_ = odom_yaw_;
      // A long sensor gap invalidates the cumulative window as well as the
      // single-sample derivative. Do not let pre-gap travel leak into the
      // next causal model feature vector.
      recent_raw_speeds_.clear();
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      sensor_fusion_encoder_history_.clear();
      sensor_fusion_recovery_window_history_.clear();
      sensor_fusion_v2_features_.reset();
      steady_mapped_speed_history_.clear();
      sensor_fusion_window_raw_speed_mps_ = 0.0;
      sensor_fusion_window_mapped_speed_mps_ = 0.0;
      return;
    }

    // JointState.position is the encoder angle exposed by the official bridge.
    // The bridge uses the same raw angle for the rear-wheel rotation.
    const double dl = wheel_radius_ * (left_angle_ - prev_left_);
    const double dr = wheel_radius_ * (right_angle_ - prev_right_);

    // Do not turn a simulator reset, encoder rollover, or dropped/replayed
    // encoder sample into metres of fictitious motion.  Re-baselining keeps
    // /odom continuous and lets the next valid pair resume integration.
    const bool paired_encoder_reset =
      dl < -encoder_reset_threshold_rad_ * wheel_radius_ &&
      dr < -encoder_reset_threshold_rad_ * wheel_radius_;
    const double source_time_step_limit_m = max_encoder_step_speed_mps_ * dt;
    const bool impossible_source_time_step =
      std::abs(dl) > source_time_step_limit_m ||
      std::abs(dr) > source_time_step_limit_m;
    if (paired_encoder_reset || std::abs(dl) > max_encoder_step_m_ ||
        std::abs(dr) > max_encoder_step_m_ || impossible_source_time_step) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Encoder discontinuity/reset (left=%.3f m right=%.3f m dt=%.4f); "
        "rebaselining only",
        dl, dr, dt);
      prev_left_ = left_angle_;
      prev_right_ = right_angle_;
      prev_stamp_ = stamp;
      recent_raw_speeds_.clear();
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      sensor_fusion_encoder_history_.clear();
      sensor_fusion_v2_features_.reset();
      steady_mapped_speed_history_.clear();
      sensor_fusion_window_raw_speed_mps_ = 0.0;
      sensor_fusion_window_mapped_speed_mps_ = 0.0;
      prev_yaw_ = odom_yaw_;
      raw_wheel_speed_mps_ = std::numeric_limits<double>::quiet_NaN();
      corrected_wheel_speed_mps_ = std::numeric_limits<double>::quiet_NaN();
      longitudinal_slip_ = std::numeric_limits<double>::quiet_NaN();
      ++encoder_reset_count_;
      encoder_reset_active_ = true;
      last_encoder_reset_stamp_ = stamp;
      publish_odom(stamp, speed_mps_);
      return;
    }

    const double wheel_distance = 0.5 * (dl + dr);
    const double wheel_speed = wheel_distance / dt;
    const double wheel_speed_abs = std::abs(wheel_speed);
    const double encoder_position_m = 0.5 * wheel_radius_ *
      (left_angle_ + right_angle_);
    const bool active_deceleration =
      imu_acceleration_filtered_mps2_ < -regime_enter_acceleration_mps2_;
    // The timestamped encoder window is a cumulative displacement estimate.
    // A braking interval is a deliberate boundary for that estimate: the
    // driven encoder may freeze while the body continues to move, so keeping
    // those samples would report a falsely low speed after throttle resumes.
    if (active_deceleration != sensor_fusion_braking_window_active_) {
      sensor_fusion_encoder_history_.clear();
      // A brake is a model-regime boundary as well as an encoder-window
      // boundary. Do not let acceleration/steady-state samples from before
      // the brake influence the first post-brake prediction.
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      sensor_fusion_window_raw_speed_mps_ = 0.0;
      sensor_fusion_window_mapped_speed_mps_ = 0.0;
      sensor_fusion_window_recovery_elapsed_s_ = 0.0;
      sensor_fusion_recovery_window_history_.clear();
      sensor_fusion_braking_window_active_ = active_deceleration;
      if (active_deceleration) {
        // The next nonzero encoder windows are not trusted merely because
        // the fixed recovery timer has elapsed. They must pass the separate
        // multi-sample reacquisition gate below.
        sensor_fusion_wheel_reacquisition_required_ = true;
      }
    } else if (active_deceleration) {
      sensor_fusion_window_recovery_elapsed_s_ = 0.0;
    } else {
      sensor_fusion_window_recovery_elapsed_s_ = std::min(
        sensor_fusion_window_recovery_s_,
        sensor_fusion_window_recovery_elapsed_s_ + std::max(0.0, dt));
    }
    const bool sensor_fusion_window_recovering =
      !active_deceleration && sensor_fusion_wheel_reacquisition_required_;
    const double window_raw_speed = sensor_fusion_window_raw_speed(
      stamp.seconds(), encoder_position_m, wheel_speed_abs);
    sensor_fusion_window_raw_speed_mps_ = window_raw_speed;
    const double window_mapped_speed =
      body_speed_from_wheel_speed(window_raw_speed);
    sensor_fusion_window_mapped_speed_mps_ = window_mapped_speed;
    const double sensor_fusion_window_age_s =
      sensor_fusion_encoder_history_.empty() ? 0.0 : std::max(
      0.0, stamp.seconds() - sensor_fusion_encoder_history_.front().first);
    // A brake boundary invalidates the old cumulative window, but the first
    // post-brake windows can still contain quantized/burst samples. Keep a
    // small, separate reacquisition history and accept it only after five
    // consecutive values agree and remain plausible relative to the IMU.
    if (!active_deceleration && sensor_fusion_wheel_reacquisition_required_ &&
      window_mapped_speed > stationary_speed_threshold_mps_ &&
      std::abs(window_mapped_speed - imu_speed_mps_) <=
      observer_speed_tolerance_mps_ + max_observer_accel_mps2_ *
      std::max(dt, 0.02))
    {
      sensor_fusion_recovery_window_history_.push_back(window_mapped_speed);
      while (sensor_fusion_recovery_window_history_.size() > 5) {
        sensor_fusion_recovery_window_history_.pop_front();
      }
    }
    const bool recovery_window_stable =
      sensor_fusion_recovery_window_history_.size() >= 5 &&
      *std::max_element(
      sensor_fusion_recovery_window_history_.begin(),
      sensor_fusion_recovery_window_history_.end()) -
      *std::min_element(
      sensor_fusion_recovery_window_history_.begin(),
      sensor_fusion_recovery_window_history_.end()) <=
      sensor_fusion_recovery_stability_mps_;
    const double recovery_window_speed = recovery_window_stable ?
      sensor_history_median_existing(sensor_fusion_recovery_window_history_) :
      std::numeric_limits<double>::quiet_NaN();
    const bool post_brake_wheel_reacquired = recovery_window_stable &&
      std::isfinite(recovery_window_speed) &&
      sensor_fusion_window_recovery_elapsed_s_ >= sensor_fusion_window_recovery_s_ &&
      std::abs(recovery_window_speed - imu_speed_mps_) <=
      observer_speed_tolerance_mps_ + max_observer_accel_mps2_ *
      std::max(dt, 0.02);
    if (post_brake_wheel_reacquired) {
      sensor_fusion_wheel_reacquisition_required_ = false;
    }
    const bool recovery_window_partially_stable =
      sensor_fusion_wheel_reacquisition_required_ &&
      sensor_fusion_recovery_window_history_.size() >= 3 &&
      *std::max_element(
      sensor_fusion_recovery_window_history_.begin(),
      sensor_fusion_recovery_window_history_.end()) -
      *std::min_element(
      sensor_fusion_recovery_window_history_.begin(),
      sensor_fusion_recovery_window_history_.end()) <=
      sensor_fusion_recovery_stability_mps_ &&
      std::isfinite(recovery_window_speed) &&
      recovery_window_speed > stationary_speed_threshold_mps_ &&
      std::abs(recovery_window_speed - imu_speed_mps_) <=
      low_speed_encoder_max_imu_gap_mps_;
    if (wheel_distance > 0.0) {
      last_motion_sign_ = 1.0;
    } else if (wheel_distance < 0.0) {
      last_motion_sign_ = -1.0;
    }
    const double mapped_speed = body_speed_from_wheel_speed(wheel_speed_abs);
    // The bridge can deliver repeated encoder positions followed by a burst
    // containing the accumulated increment.  The instantaneous derivative is
    // therefore a useful diagnostic, but it is not a stable velocity
    // measurement.  Use the timestamped travel window for fusion decisions
    // and pose integration so the repeated zero/burst pattern is averaged in
    // time rather than interpreted as alternating stop and wheel spin.
    const double fusion_wheel_speed = window_raw_speed;
    const double fusion_mapped_speed = window_mapped_speed;
    // Encoder callbacks can be ahead of the latest IMU callback by one
    // bridge burst. Keep the observer at its IMU timestamp, but extrapolate
    // only the speed used for this pair output to the pair timestamp. This
    // avoids publishing a one-burst-old braking speed without advancing the
    // observer clock and integrating the same interval again later.
    double imu_pair_speed_mps = imu_speed_mps_;
    double imu_pair_lead_s = 0.0;
    if (imu_speed_ready_ && have_imu_speed_stamp_) {
      imu_pair_lead_s = (stamp - last_imu_speed_stamp_).seconds();
      if (imu_pair_lead_s > 1.0e-4 &&
        imu_pair_lead_s <= imu_pair_extrapolation_max_s_)
      {
        const double observer_acceleration =
          imu_observer_acceleration_mps2_ - longitudinal_observer_.bias();
        imu_pair_speed_mps = std::clamp(
          imu_pair_speed_mps + observer_acceleration * imu_pair_lead_s,
          0.0, 30.0);
      }
    }
    imu_pair_speed_mps_ = imu_pair_speed_mps;
    imu_pair_lead_s_ = std::max(0.0, imu_pair_lead_s);
    const double imu_body_speed = std::clamp(imu_pair_speed_mps, 0.0, 30.0);
    const double slip_denominator = std::max(imu_body_speed, 0.25);
    const double imu_longitudinal_slip =
      (fusion_wheel_speed - imu_body_speed) / slip_denominator;
    const double absolute_longitudinal_slip = std::abs(imu_longitudinal_slip);
    raw_wheel_speed_mps_ = wheel_speed;
    corrected_wheel_speed_mps_ = std::copysign(mapped_speed, wheel_speed);
    const bool slip_observable = imu_speed_ready_ &&
      imu_body_speed > slip_observation_min_speed_mps_;
    const bool high_slip = slip_observable &&
      absolute_longitudinal_slip >= forward_extremum_slip_;
    const bool asymptotic_slip = high_slip &&
      absolute_longitudinal_slip >= forward_asymptote_slip_;
    // The documented force curve is not a velocity inversion.  It does,
    // however, provide a physically meaningful confidence transition: once
    // the tire force falls from the extremum to the asymptote, the driven
    // wheel angle carries progressively less information about body speed.
    // Use that transition as a wheel-confidence feature for the fitted
    // sensor-only fusion model and as the conservative fallback weight when
    // the learned model is disabled.
    const double wheel_observation_confidence = slip_observable ?
      longitudinal_wheel_observation_confidence(absolute_longitudinal_slip) : 1.0;
    const double previous_fusion_mapped_speed = sensor_history_median_existing(
      sensor_fusion_mapped_history_);
    const bool low_speed_encoder_window_stable = sensor_history_stable(
      sensor_fusion_mapped_history_, fusion_mapped_speed,
      low_speed_encoder_stability_mps_, low_speed_encoder_stability_samples_);
    const bool low_speed_encoder_window_mature =
      sensor_fusion_window_age_s >= sensor_fusion_window_low_speed_duration_s_;
    const bool low_speed_encoder_priority = imu_speed_ready_ &&
      fusion_mapped_speed > stationary_speed_threshold_mps_ &&
      fusion_mapped_speed <= low_speed_encoder_priority_mps_ &&
      std::isfinite(previous_fusion_mapped_speed) &&
      low_speed_encoder_window_stable &&
      low_speed_encoder_window_mature &&
      std::abs(fusion_mapped_speed - imu_body_speed) <=
      low_speed_encoder_max_imu_gap_mps_ &&
      !sensor_fusion_window_recovering &&
      !sensor_fusion_wheel_reacquisition_required_;
    const bool wheel_encoder_priority = low_speed_encoder_priority ||
      post_brake_wheel_reacquired;
    // The force-curve slip gate is useful once wheel speed is well observed,
    // but near standstill it confuses accumulated IMU bias with wheel slip.
    // A stable low-speed encoder window is therefore allowed to correct the
    // observer even when the ratio computed against the biased IMU integral
    // crosses the asymptotic threshold.
    const bool map_observation_rejected = asymptotic_slip &&
      !wheel_encoder_priority;
    const double effective_wheel_observation_confidence =
      wheel_encoder_priority ? 1.0 : wheel_observation_confidence;
    const double slip_threshold = std::max(
      wheel_slip_threshold_mps_,
      wheel_slip_ratio_ * std::max(fusion_wheel_speed, imu_body_speed));
    const bool slip_map_active = fusion_wheel_speed >= wheel_slip_activation_speed_mps_;
    const bool frozen_motion_candidate = imu_speed_ready_ &&
      fusion_wheel_speed <= stationary_speed_threshold_mps_ &&
      imu_body_speed > stationary_speed_threshold_mps_;
    if (imu_speed_ready_) {
      update_motion_regime(frozen_motion_candidate, dt);
    }
    // A throttle downshift can make the driven wheel speed change before the
    // body has decelerated. Hold that wheel observation out of the observer
    // for a short causal interval instead of treating the new wheel speed as
    // the current body speed.
    const double previous_fusion_imu_speed = sensor_history_median_existing(
      sensor_fusion_imu_history_);
    const double wheel_window_drop = std::isfinite(previous_fusion_mapped_speed) ?
      previous_fusion_mapped_speed - fusion_mapped_speed : 0.0;
    const double imu_body_speed_drop = std::isfinite(previous_fusion_imu_speed) ?
      previous_fusion_imu_speed - imu_body_speed : 0.0;
    const bool wheel_downshift_candidate = imu_speed_ready_ &&
      std::isfinite(previous_fusion_mapped_speed) &&
      std::isfinite(previous_fusion_imu_speed) &&
      wheel_window_drop >= sensor_fusion_wheel_downshift_min_drop_mps_ &&
      wheel_window_drop > imu_body_speed_drop +
      sensor_fusion_wheel_downshift_min_drop_mps_ * 0.25 &&
      fusion_mapped_speed > stationary_speed_threshold_mps_ &&
      fusion_mapped_speed < imu_body_speed - 0.15 &&
      imu_acceleration_filtered_mps2_ <= regime_exit_acceleration_mps2_;
    if (wheel_downshift_candidate) {
      sensor_fusion_wheel_downshift_hold_remaining_s_ = std::max(
        sensor_fusion_wheel_downshift_hold_remaining_s_,
        sensor_fusion_wheel_downshift_hold_s_);
      // The current sample becomes the first post-shift sample; do not let
      // pre-shift values make it look like a stable new wheel measurement.
      sensor_fusion_raw_history_.clear();
      sensor_fusion_mapped_history_.clear();
      sensor_fusion_imu_history_.clear();
      sensor_fusion_gap_history_.clear();
      steady_mapped_speed_history_.clear();
    }
    const bool wheel_downshift_transient =
      sensor_fusion_wheel_downshift_hold_remaining_s_ > 0.0;
    const bool deceleration_encoder_untrusted = active_deceleration ||
      motion_regime_ == SensorMotionRegime::DECELERATING ||
      sensor_fusion_wheel_reacquisition_required_ ||
      wheel_downshift_transient;
    double fused_speed = fusion_mapped_speed;
    bool encoder_dropout_hold = false;
    sensor_fusion_model_active_ = false;
    sensor_fusion_model_speed_mps_ = 0.0;
    sensor_fusion_model_spread_mps_ = 0.0;
    sensor_fusion_v2_global_speed_mps_ = 0.0;
    sensor_fusion_v2_base_speed_mps_ = 0.0;
    sensor_fusion_v2_braking_speed_mps_ = 0.0;
    sensor_fusion_v2_braking_blend_ = 0.0;
    sensor_fusion_v2_global_residual_ = 0.0;
    sensor_fusion_v2_braking_residual_ = 0.0;
    sensor_fusion_v2_active_ = false;
    if (fusion_wheel_speed <= stationary_speed_threshold_mps_) {
      // A zero/short encoder window is not a stationary observation when the
      // IMU is actively accelerating or braking.  In particular, after a
      // diagnostic reset the vehicle can begin moving before the first
      // timestamped encoder window becomes nonzero.  Do not carry the prior
      // quiet interval into that launch and never let it satisfy the stop
      // gate.  The counter resumes only after acceleration has become quiet.
      const bool imu_acceleration_quiet =
        std::abs(imu_acceleration_robust_mps2_) <=
        imu_stationary_acceleration_threshold_mps2_;
      if (!imu_acceleration_quiet) {
        zero_encoder_duration_s_ = 0.0;
      } else {
        zero_encoder_duration_s_ += std::max(0.0, dt);
      }
      // The simulator may freeze the driven-wheel encoder during a brake or
      // passive coast. Continue with the independent IMU prediction while it
      // is moving; only declare a stop after speed and acceleration have
      // stayed near zero for the confirmation interval. No command or
      // actuator feedback is required by this estimator.
      const bool imu_stop_confirmed =
        zero_encoder_duration_s_ >= zero_encoder_stop_confirm_sec_ &&
        imu_acceleration_quiet &&
        imu_speed_mps_ <= imu_stop_speed_threshold_mps_;
      if (!imu_stop_confirmed && imu_body_speed > stationary_speed_threshold_mps_) {
        // A frozen driven wheel has two observable braking intervals. During
        // active braking the IMU remains primary, but the calibrated prior may
        // correct only a positive deceleration residual. After the IMU goes
        // quiet, the same prior models the remaining slip before stop
        // confirmation. In neither case is the speed reset to zero early.
        const bool frozen_brake_prior_allowed =
          frozen_encoder_model_enabled_ &&
          ((!imu_acceleration_quiet && active_deceleration) ||
          (imu_acceleration_quiet &&
          zero_encoder_duration_s_ >= frozen_encoder_model_delay_sec_));
        if (frozen_brake_prior_allowed) {
          // The fit is deliberately a deceleration prior, not a velocity
          // reset. It keeps the car moving for the modelled remaining slip
          // distance and then lets the normal stop confirmation gate close.
          frozen_encoder_model_decel_mps2_ = std::clamp(
            frozen_encoder_decel_intercept_mps2_ +
            frozen_encoder_decel_speed_gain_per_s_ * imu_body_speed,
            0.0, frozen_encoder_decel_max_mps2_);
          const double measured_deceleration = std::max(
            0.0, -imu_observer_acceleration_mps2_);
          const double positive_correction = std::max(
            0.0, frozen_encoder_model_decel_mps2_ - measured_deceleration);
          longitudinal_observer_.add_speed_delta(
            -frozen_encoder_braking_model_blend_ * positive_correction * dt);
          imu_speed_mps_ = longitudinal_observer_.speed();
          frozen_encoder_model_active_ = true;
        }
        fused_speed = frozen_encoder_model_active_ ?
          std::clamp(imu_speed_mps_, 0.0, 30.0) : imu_body_speed;
        encoder_dropout_hold = true;
      } else {
        fused_speed = 0.0;
        longitudinal_observer_.set_speed(0.0);
        imu_speed_mps_ = longitudinal_observer_.speed();
      }
      // A frozen wheel at non-zero body speed has Sx approximately -1.0,
      // which is beyond the documented asymptote. Mark it as uncertain so
      // the downstream EKF gives AMCL more authority over the pose.
      wheel_slip_detected_ = asymptotic_slip || encoder_dropout_hold;
      wheel_observation_confidence_ = encoder_dropout_hold ? 0.0 : 1.0;
    } else if (imu_speed_ready_) {
      zero_encoder_duration_s_ = 0.0;
      // The calibrated map converts the documented encoder geometry into a
      // body-speed measurement, including the measured driven-wheel slip.
      // Use it as an absolute correction only when it is physically plausible
      // relative to the IMU prediction. Within the calibrated operating
      // envelope the map and IMU should agree; under a spinning-wheel burst
      // the map can be much higher than body speed and must be rejected.
      const double wheel_imu_gap = fusion_wheel_speed - imu_body_speed;
      wheel_slip_detected_ = slip_map_active &&
        (high_slip || wheel_imu_gap > slip_threshold);
      const double max_map_prediction_gap = observer_speed_tolerance_mps_ +
        max_observer_accel_mps2_ * dt;
      // During real braking the driven wheel does not measure body speed:
      // its angular momentum and tire slip keep the cumulative wheel window
      // above the car's speed, then the encoder can freeze altogether.  The
      // IMU prediction is the only timely longitudinal measurement in this
      // regime.  Reject the wheel update as soon as the filtered IMU sees a
      // meaningful negative acceleration, without waiting for the hysteresis
      // dwell to relabel the regime.  This prevents a stale wheel window from
      // feeding speed back into the observer and delaying the deceleration.
      const bool mapped_window_stable =
        sensor_fusion_mapped_history_.size() >= 3 &&
        std::abs(fusion_mapped_speed - previous_fusion_mapped_speed) <=
        sensor_fusion_window_stability_mps_ &&
        !sensor_fusion_wheel_reacquisition_required_;
      if (!map_observation_rejected && !deceleration_encoder_untrusted &&
        (wheel_encoder_priority || mapped_window_stable) &&
        std::abs(fusion_mapped_speed - imu_body_speed) <= max_map_prediction_gap) {
        // The scalar observer determines the correction from its covariance;
        // regime/slip confidence enters as measurement variance. This avoids
        // using one fixed blend at launch, steady speed, and braking.
        const double measurement_gate = std::max(
          0.05, std::min(observer_innovation_gate_mps_, max_map_prediction_gap));
        const double measurement_variance = post_brake_wheel_reacquired ?
          observer_steady_measurement_variance_m2ps2_ :
          observer_measurement_variance(wheel_observation_confidence);
        longitudinal_observer_.update(
          post_brake_wheel_reacquired ? recovery_window_speed : fusion_mapped_speed,
          measurement_variance, measurement_gate);
        imu_speed_mps_ = longitudinal_observer_.speed();
        fused_speed = imu_speed_mps_;
      } else {
        fused_speed = imu_body_speed;
        // Do not repeatedly compare the same delayed encoder burst with the
        // IMU prediction on the next callback.  The next valid pair is the
        // first opportunity to resynchronize the absolute wheel measurement.
        prev_left_ = left_angle_;
        prev_right_ = right_angle_;
        prev_stamp_ = stamp;
        recent_raw_speeds_.clear();
      }
      wheel_observation_confidence_ = deceleration_encoder_untrusted ? 0.0 :
        effective_wheel_observation_confidence;
      wheel_slip_detected_ = wheel_slip_detected_ ||
        deceleration_encoder_untrusted;

      // The driven encoder can stop immediately when braking while the body
      // continues to move. In that interval the IMU is the only timely
      // measurement, but the calibration data shows that its longitudinal
      // acceleration is slightly less negative than the actual body
      // deceleration. Apply the speed-dependent prior as a bounded correction
      // only when the encoder is already uninformative. This keeps the IMU in
      // the loop and avoids interpreting a zero wheel increment as a stop.
      const bool braking_encoder_uninformative = active_deceleration &&
        (encoder_dropout_hold || wheel_speed_abs <= stationary_speed_threshold_mps_ ||
        fusion_wheel_speed <= stationary_speed_threshold_mps_);
      if (frozen_encoder_model_enabled_ && braking_encoder_uninformative &&
        imu_body_speed > stationary_speed_threshold_mps_)
      {
        const double prior_deceleration = std::clamp(
          frozen_encoder_decel_intercept_mps2_ +
            frozen_encoder_decel_speed_gain_per_s_ * imu_body_speed,
          0.0, frozen_encoder_decel_max_mps2_);
        const double measured_deceleration = std::max(
          0.0, -imu_observer_acceleration_mps2_);
        const double positive_correction = std::max(
          0.0, prior_deceleration - measured_deceleration);
        const double correction = frozen_encoder_braking_model_blend_ *
          positive_correction;
        if (correction > 0.0) {
          longitudinal_observer_.add_speed_delta(-correction * dt);
          imu_speed_mps_ = longitudinal_observer_.speed();
          fused_speed = imu_speed_mps_;
          frozen_encoder_model_active_ = true;
          frozen_encoder_model_decel_mps2_ = prior_deceleration;
        }
      }
    } else {
      wheel_slip_detected_ = false;
      wheel_observation_confidence_ = 1.0;
    }

    // Before the full post-brake wheel gate passes, a short sequence of
    // tightly clustered windows is already useful as a bounded correction.
    // Apply only an upward correction toward the window median: the IMU can
    // be left below the true speed after integrating the brake, while a
    // premature downward wheel correction would recreate the encoder-freeze
    // failure. This does not update the observer state and cannot clear the
    // full reacquisition lock.
    if (recovery_window_partially_stable &&
      recovery_window_speed > fused_speed)
    {
      const double correction = std::min(
        sensor_fusion_recovery_encoder_max_correction_mps_,
        sensor_fusion_recovery_encoder_blend_ *
        (recovery_window_speed - fused_speed));
      fused_speed += correction;
    }

    // The v2 model is fitted offline against simulator truth but is causal at
    // runtime: it only sees the independent IMU observer, encoder-derived
    // speeds, timing, acceleration, and their causal history. Ground truth,
    // throttle, IPS, and simulator odometry never enter this path.
    const double model_imu_speed = std::clamp(imu_body_speed, 0.0, 30.0);
    const bool frozen_motion = encoder_dropout_hold &&
      model_imu_speed > stationary_speed_threshold_mps_;
    const bool model_motion =
      fusion_wheel_speed > stationary_speed_threshold_mps_ || frozen_motion;
    if (imu_speed_ready_ && model_motion && frozen_motion) {
      motion_regime_ = SensorMotionRegime::FROZEN;
    }
    const bool learned_steady_model_allowed =
      sensor_fusion_steady_model_enabled_ ||
      motion_regime_ != SensorMotionRegime::STEADY;
    // A frozen driven-wheel encoder is ambiguous: during a launch it is a
    // delayed observation, while during braking it carries no body-speed
    // information.  The offline fit keeps these cases separate.  Select the
    // launch branch only from the causal filtered IMU acceleration; the
    // externally reported motion regime remains FROZEN for compatibility.
    SensorMotionRegime model_regime = motion_regime_;
    if (motion_regime_ == SensorMotionRegime::FROZEN &&
      imu_acceleration_filtered_mps2_ >= -regime_acceleration_threshold_mps2_)
    {
      model_regime = SensorMotionRegime::FROZEN_ACCELERATING;
    }
    if (sensor_fusion_v2_enabled_ && imu_speed_ready_) {
      // The v2 baseline is the current pre-v2 causal sensor estimate. Using
      // speed_mps_ here feeds the previous v2 output back into its own next
      // prediction, which was not present in the attached training replay
      // and can accumulate a low-speed/high-slip bias on a fresh run.
      const double v2_base_speed = std::clamp(fused_speed, 0.0, 30.0);
      sensor_fusion_v2_base_speed_mps_ = v2_base_speed;
      const auto v2_features = sensor_fusion_v2_features_.update(
        v2_base_speed, imu_pair_speed_mps, imu_speed_mps_, wheel_speed_abs,
        mapped_speed, window_raw_speed, window_mapped_speed,
        imu_observer_acceleration_mps2_, longitudinal_observer_.bias(),
        std::clamp(wheel_observation_confidence_, 0.0, 1.0),
        imu_raw_acceleration_mps2_, imu_lateral_acceleration_mps2_,
        imu_yaw_rate_, stamp.seconds(), dt);

      if (model_motion) {
        const auto prediction = sensor_fusion_v2_model_.predict(v2_features);
        const double global_speed = std::clamp(
          v2_base_speed + std::max(v2_base_speed, 0.5) *
          prediction.global_fractional_residual, 0.0, 30.0);
        const double braking_speed = std::clamp(
          v2_base_speed + prediction.braking_absolute_residual_mps,
          0.0, 30.0);
        const double braking_blend = std::clamp(
          (-imu_observer_acceleration_mps2_ -
          sensor_fusion_v2_braking_blend_offset_mps2_) /
          sensor_fusion_v2_braking_blend_range_mps2_, 0.0, 1.0);
        const double v2_speed = (1.0 - braking_blend) * global_speed +
          braking_blend * braking_speed;
        if (prediction.valid && std::isfinite(v2_speed)) {
          fused_speed = v2_speed;
          sensor_fusion_model_active_ = true;
          sensor_fusion_model_speed_mps_ = v2_speed;
          sensor_fusion_model_spread_mps_ =
            std::abs(global_speed - braking_speed);
          sensor_fusion_v2_global_speed_mps_ = global_speed;
          sensor_fusion_v2_braking_speed_mps_ = braking_speed;
          sensor_fusion_v2_braking_blend_ = braking_blend;
          sensor_fusion_v2_global_residual_ =
            prediction.global_fractional_residual;
          sensor_fusion_v2_braking_residual_ =
            prediction.braking_absolute_residual_mps;
          sensor_fusion_v2_active_ = true;
        }
      }
    } else if (sensor_fusion_model_enabled_ && imu_speed_ready_ &&
      model_motion && learned_steady_model_allowed &&
      // Braking gets a separate conservative blend below. Do not let the
      // generic branch replace the observer during a brake or wheel
      // reacquisition transient. The first post-brake window is explicitly
      // treated as untrusted until the separate five-sample gate passes.
      !active_deceleration && !sensor_fusion_window_recovering &&
      !wheel_downshift_transient &&
      motion_regime_ != SensorMotionRegime::DECELERATING)
    {
      const auto features = sensor_fusion_features(
        wheel_speed, mapped_speed, model_imu_speed, dt, pair_skew,
        effective_wheel_observation_confidence, window_raw_speed,
        longitudinal_observer_.bias(), imu_observer_acceleration_mps2_);
      const auto prediction = sensor_fusion_model_.predict(model_regime, features);
      sensor_fusion_model_speed_mps_ = prediction.speed_mps;
      sensor_fusion_model_spread_mps_ = prediction.spread_mps;
      const bool prediction_finite = prediction.valid &&
        std::isfinite(prediction.speed_mps) &&
        std::isfinite(prediction.spread_mps);
      const bool prediction_bounded = prediction.spread_mps <=
        regime_model_max_spread_mps_ &&
        std::abs(prediction.speed_mps - fused_speed) <=
        regime_model_max_baseline_delta_mps_;
      // A frozen driven-wheel encoder is especially ambiguous during a
      // launch after a target downshift.  In that state the scalar observer
      // is already the causal IMU prediction; a learned branch trained on
      // stop-to-throttle launches must not pull it down merely because its
      // wheel features look like a delayed zero.  Keep upward corrections,
      // but make this branch one-sided so a stale frozen wheel cannot create
      // an artificial loss of body speed.
      const bool frozen_launch_downward_correction =
        model_regime == SensorMotionRegime::FROZEN_ACCELERATING &&
        prediction.speed_mps < fused_speed;
      if (prediction_finite && prediction_bounded) {
        const bool transient_model = frozen_motion;
        if (transient_model) {
          // During a frozen/post-brake interval the model is useful only for
          // recovering an IMU estimate that is known to lag the body. Never
          // pull the estimate down and cap the upward correction so a forest
          // extrapolation cannot recreate the previous overshoot failure.
          const double upward_correction = std::max(
            0.0, prediction.speed_mps - fused_speed);
          double transient_max_correction =
            sensor_fusion_transient_model_max_correction_mps_;
          if (model_imu_speed < sensor_fusion_transient_low_speed_threshold_mps_) {
            transient_max_correction = std::min(
              transient_max_correction,
              sensor_fusion_transient_low_speed_max_correction_mps_);
          } else if (model_imu_speed < sensor_fusion_transient_mid_speed_threshold_mps_) {
            transient_max_correction = std::min(
              transient_max_correction,
              sensor_fusion_transient_mid_speed_max_correction_mps_);
          }
          const double bounded_correction = std::min(
            transient_max_correction,
            sensor_fusion_transient_model_blend_ * upward_correction);
          if (bounded_correction > 0.0) {
            sensor_fusion_model_active_ = true;
            fused_speed += bounded_correction;
          }
        } else if (!frozen_launch_downward_correction) {
          sensor_fusion_model_active_ = true;
          // The learned prediction corrects a causal IMU/encoder baseline;
          // it is not an independent measurement. A partial correction is
          // more stable at regime boundaries and prevents a tree leaf trained
          // on a sparse low-speed burst from replacing the observer outright.
          fused_speed += sensor_fusion_model_blend_ * (
            prediction.speed_mps - fused_speed);
        }
      }
    }

    // Only collect absolute wheel/map observations while the observer is in
    // the steady regime. A launch or braking sample at the same speed is not
    // interchangeable with a constant-speed sample and must not be allowed
    // to re-anchor the IMU state later.
    // The IMU-integrated observer can retain a positive speed bias after a
    // hard launch. Once longitudinal acceleration is quiet, a stable mapped
    // wheel observation below that observer is the useful absolute speed
    // reference. This is deliberately one-sided: a wheel speed above the
    // IMU remains subject to the documented driven-wheel slip gates. The
    // short history also prevents one delayed/quantized encoder sample from
    // becoming a new observer state.
    // Calculate the reference from prior steady observations. Including the
    // current sample twice would make one low encoder packet look like a
    // stable speed and could re-anchor the observer to that outlier.
    const double steady_mapped_speed = sensor_history_median_existing(
      steady_mapped_speed_history_);
    const bool steady_wheel_consistent = std::isfinite(steady_mapped_speed) &&
      std::abs(fusion_mapped_speed - steady_mapped_speed) <=
      observer_speed_tolerance_mps_ + max_observer_accel_mps2_ * dt;
    const bool steady_encoder_reanchor = steady_encoder_reanchor_enabled_ &&
      motion_regime_ == SensorMotionRegime::STEADY &&
      !encoder_dropout_hold &&
      !sensor_fusion_window_recovering &&
      !wheel_downshift_transient &&
      steady_mapped_speed_history_.size() >= 2 &&
      low_speed_encoder_window_mature &&
      std::abs(imu_acceleration_robust_mps2_) <=
      regime_acceleration_threshold_mps2_ &&
      fusion_mapped_speed > stationary_speed_threshold_mps_ &&
      steady_mapped_speed > stationary_speed_threshold_mps_ &&
      // At low and moderate speed the calibrated wheel/map observation is
      // the useful absolute velocity reference.  The old one-sided gate
      // only accepted a wheel value below the IMU observer, so a launch bias
      // could leave /odom permanently low even after the car had settled.
      // Reject large disagreements here; the slip-confidence gate below
      // keeps wheel-spin bursts from becoming a new steady reference.
      std::abs(steady_mapped_speed - model_imu_speed) <=
      observer_speed_tolerance_mps_ &&
      effective_wheel_observation_confidence >= 0.75 &&
      sensor_history_stable(
        steady_mapped_speed_history_, fusion_mapped_speed,
        low_speed_encoder_stability_mps_, 5) &&
      steady_wheel_consistent;
    if (steady_encoder_reanchor) {
      longitudinal_observer_.set_speed(steady_mapped_speed);
      imu_speed_mps_ = longitudinal_observer_.speed();
      fused_speed = imu_speed_mps_;
      wheel_slip_detected_ = false;
      wheel_observation_confidence_ = 1.0;
    }

    if (motion_regime_ != SensorMotionRegime::STEADY) {
      steady_mapped_speed_history_.clear();
    } else if (fusion_mapped_speed > stationary_speed_threshold_mps_ &&
      effective_wheel_observation_confidence >= 0.75 &&
      !sensor_fusion_window_recovering) {
      // Keep low-confidence wheel-spin samples out of the steady reference
      // history.  Otherwise a single driven-wheel burst can poison the
      // median for the next several native telemetry cycles.
      steady_mapped_speed_history_.push_back(fusion_mapped_speed);
      while (steady_mapped_speed_history_.size() > 5) {
        steady_mapped_speed_history_.pop_front();
      }
    }

    fused_speed = std::clamp(fused_speed, 0.0, 30.0);
    sensor_fusion_wheel_downshift_hold_remaining_s_ = std::max(
      0.0, sensor_fusion_wheel_downshift_hold_remaining_s_ - dt);
    // Report slip against the final allowed body-speed estimate. The IMU
    // innovation still controls wheel trust; this diagnostic is less
    // sensitive to IMU integration lag than the raw IMU ratio alone.
    longitudinal_slip_ = (fusion_wheel_speed - fused_speed) /
      std::max(std::abs(fused_speed), 0.25);
    // Integrate the observer's body speed for both pose and twist. Raw wheel
    // distance is not a body-distance increment while driven-wheel slip is
    // present; yaw still comes from the allowed IMU gyro/orientation.
    // Do not integrate a repeated zero encoder position as motion.  Only the
    // explicitly bounded dropout hold may use the IMU speed when the wheel
    // delta is zero.
    const double ds = (fusion_wheel_speed > stationary_speed_threshold_mps_ ||
      encoder_dropout_hold) ?
      std::copysign(fused_speed * dt, wheel_distance == 0.0 ?
      (last_motion_sign_ >= 0.0 ? 1.0 : -1.0) : wheel_distance) : 0.0;

    remember_sensor_fusion_sample(
      fusion_wheel_speed, fusion_mapped_speed, imu_body_speed,
      std::abs(fusion_mapped_speed - imu_body_speed));
    remember_sensor_fusion_encoder_sample(stamp.seconds(), encoder_position_m);

    const double dyaw = wrap_angle(odom_yaw_ - prev_yaw_);
    const double yaw_mid = prev_yaw_ + 0.5 * dyaw;
    x_ += ds * std::cos(yaw_mid);
    y_ += ds * std::sin(yaw_mid);

    prev_left_ = left_angle_;
    prev_right_ = right_angle_;
    prev_stamp_ = stamp;
    prev_yaw_ = odom_yaw_;
    // The fusion path has already rejected instantaneous encoder bursts with
    // its timestamped window, slip gate, observer covariance, and learned
    // regime model. Applying a second median here creates an avoidable output
    // delay at a brake boundary and makes the speed controller react to an
    // old /odom value. Publish the accepted fused estimate directly; the
    // estimator's causal filters remain upstream of this point.
    speed_mps_ = fused_speed;
    recent_raw_speeds_.clear();
    publish_odom(stamp, speed_mps_);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Odometry: dl=%.3f dr=%.3f ds=%.3f wheel_speed=%.3f mapped=%.3f body_speed=%.3f "
      "speed=%.3f pose=(%.3f, %.3f, %.3f) enc=(%.3f, %.3f) wheel_slip=%s confidence=%.2f",
      dl, dr, ds, wheel_speed_abs, mapped_speed, fused_speed, speed_mps_, x_, y_, odom_yaw_,
      left_angle_, right_angle_,
      wheel_slip_detected_ ? "true" : "false", wheel_observation_confidence);
  }

  void reset_longitudinal_observer()
  {
    longitudinal_observer_.reset();
    imu_speed_mps_ = 0.0;
    sensor_fusion_wheel_downshift_hold_remaining_s_ = 0.0;
    imu_acceleration_filtered_mps2_ = 0.0;
    imu_acceleration_robust_mps2_ = 0.0;
    imu_observer_acceleration_mps2_ = 0.0;
    imu_acceleration_history_.clear();
    imu_speed_ready_ = false;
    have_imu_speed_stamp_ = false;
    zero_encoder_duration_s_ = 0.0;
    wheel_slip_detected_ = false;
    wheel_observation_confidence_ = 1.0;
    frozen_encoder_model_active_ = false;
    frozen_encoder_model_decel_mps2_ = 0.0;
    imu_pair_speed_mps_ = 0.0;
    imu_pair_lead_s_ = 0.0;
    motion_regime_ = SensorMotionRegime::STEADY;
    regime_candidate_ = SensorMotionRegime::STEADY;
    regime_candidate_dwell_s_ = 0.0;
  }

  double observer_measurement_variance(double confidence) const
  {
    double variance = observer_encoder_measurement_variance_m2ps2_;
    switch (motion_regime_) {
      case SensorMotionRegime::ACCELERATING:
        variance = observer_accelerating_measurement_variance_m2ps2_;
        break;
      case SensorMotionRegime::STEADY:
        variance = observer_steady_measurement_variance_m2ps2_;
        break;
      case SensorMotionRegime::DECELERATING:
        variance = observer_decelerating_measurement_variance_m2ps2_;
        break;
      case SensorMotionRegime::FROZEN:
        variance = observer_encoder_measurement_variance_m2ps2_ * 100.0;
        break;
      case SensorMotionRegime::FROZEN_ACCELERATING:
        variance = observer_encoder_measurement_variance_m2ps2_ * 100.0;
        break;
    }
    const double usable_confidence = std::clamp(confidence, 0.05, 1.0);
    return variance / (usable_confidence * usable_confidence);
  }

  void update_motion_regime(bool frozen, double dt)
  {
    if (frozen) {
      motion_regime_ = SensorMotionRegime::FROZEN;
      regime_candidate_ = SensorMotionRegime::FROZEN;
      regime_candidate_dwell_s_ = 0.0;
      return;
    }

    const SensorMotionRegime candidate = classify_motion_regime();
    if (candidate == motion_regime_) {
      regime_candidate_ = candidate;
      regime_candidate_dwell_s_ = 0.0;
      return;
    }
    if (candidate != regime_candidate_) {
      regime_candidate_ = candidate;
      regime_candidate_dwell_s_ = std::max(0.0, dt);
    } else {
      regime_candidate_dwell_s_ += std::max(0.0, dt);
    }
    if (regime_candidate_dwell_s_ >= regime_min_dwell_s_) {
      motion_regime_ = candidate;
      regime_candidate_dwell_s_ = 0.0;
    }
  }

  double body_speed_from_wheel_speed(double wheel_speed) const
  {
    const double speed = std::max(0.0, wheel_speed);
    if (speed <= wheel_speed_map_wheel_mps_.front()) {
      return wheel_speed_map_body_mps_.front();
    }
    if (speed >= wheel_speed_map_wheel_mps_.back()) {
      return wheel_speed_map_body_mps_.back();
    }
    const auto upper = std::upper_bound(
      wheel_speed_map_wheel_mps_.begin(), wheel_speed_map_wheel_mps_.end(), speed);
    const size_t upper_index = static_cast<size_t>(
      std::distance(wheel_speed_map_wheel_mps_.begin(), upper));
    const size_t lower_index = upper_index - 1;
    const double lower_speed = wheel_speed_map_wheel_mps_[lower_index];
    const double upper_speed = wheel_speed_map_wheel_mps_[upper_index];
    const double ratio = (speed - lower_speed) / (upper_speed - lower_speed);
    return wheel_speed_map_body_mps_[lower_index] + ratio * (
      wheel_speed_map_body_mps_[upper_index] -
      wheel_speed_map_body_mps_[lower_index]);
  }

  double longitudinal_wheel_observation_confidence(double absolute_slip) const
  {
    if (!std::isfinite(absolute_slip) ||
      absolute_slip >= forward_asymptote_slip_)
    {
      return 0.0;
    }
    if (absolute_slip <= forward_extremum_slip_) {
      return 1.0;
    }

    // Hermite interpolation of the documented force values with zero slope
    // at both knots.  Normalize the force fall so confidence is one at the
    // extremum and zero at the asymptote.
    const double span = forward_asymptote_slip_ - forward_extremum_slip_;
    const double t = std::clamp(
      (absolute_slip - forward_extremum_slip_) / span, 0.0, 1.0);
    const double smoothstep = t * t * (3.0 - 2.0 * t);
    const double force = forward_extremum_value_ +
      (forward_asymptote_value_ - forward_extremum_value_) * smoothstep;
    return std::clamp(
      (force - forward_asymptote_value_) /
      (forward_extremum_value_ - forward_asymptote_value_), 0.0, 1.0);
  }

  static double sensor_history_median(
    const std::deque<double> & history, double current)
  {
    std::vector<double> values;
    constexpr size_t kHistoryWindow = 9;
    const size_t first = history.size() > kHistoryWindow ?
      history.size() - kHistoryWindow : 0;
    values.reserve(history.size() - first + 1);
    for (size_t index = first; index < history.size(); ++index) {
      values.push_back(history[index]);
    }
    values.push_back(current);
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  }

  static double sensor_history_median_existing(
    const std::deque<double> & history)
  {
    if (history.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> values(history.begin(), history.end());
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  }

  static bool sensor_history_stable(
    const std::deque<double> & history, double current, double max_range,
    std::size_t minimum_samples)
  {
    if (!std::isfinite(current) || history.size() < minimum_samples) {
      return false;
    }
    double minimum = current;
    double maximum = current;
    for (const double value : history) {
      if (!std::isfinite(value)) {
        return false;
      }
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
    return maximum - minimum <= std::max(0.0, max_range);
  }

  std::array<double, f1tenth_localization::SensorFusionModel::kFeatureCount>
  sensor_fusion_features(
    double wheel_speed, double mapped_speed, double imu_speed, double dt,
    double pair_skew, double confidence, double window_raw_speed,
    double imu_acceleration_bias, double imu_observer_acceleration) const
  {
    const double raw = std::abs(wheel_speed);
    const double mapped = std::abs(mapped_speed);
    const double imu = std::max(0.0, imu_speed);
    const double signed_gap = mapped - imu;
    const double gap = std::abs(signed_gap);
    const double previous_raw = sensor_fusion_raw_history_.empty() ?
      raw : sensor_fusion_raw_history_.back();
    const double previous_mapped = sensor_fusion_mapped_history_.empty() ?
      mapped : sensor_fusion_mapped_history_.back();
    const double previous_imu = sensor_fusion_imu_history_.empty() ?
      imu : sensor_fusion_imu_history_.back();
    std::array<double, f1tenth_localization::SensorFusionModel::kFeatureCount>
      features{};
    features[0] = imu;
    features[1] = mapped;
    features[2] = raw;
    features[3] = signed_gap;
    features[4] = gap;
    features[5] = std::max(0.0, confidence);
    features[6] = imu_raw_acceleration_mps2_;
    features[7] = imu_acceleration_filtered_mps2_;
    features[8] = std::abs(imu_lateral_acceleration_mps2_);
    features[9] = std::abs(imu_yaw_rate_);
    features[10] = std::max(0.0, dt);
    features[11] = std::max(0.0, pair_skew);
    features[12] = raw <= stationary_speed_threshold_mps_ ? 1.0 : 0.0;
    features[13] = sensor_history_median(sensor_fusion_raw_history_, raw);
    features[14] = sensor_history_median(sensor_fusion_mapped_history_, mapped);
    features[15] = sensor_history_median(sensor_fusion_imu_history_, imu);
    features[16] = sensor_history_median(sensor_fusion_gap_history_, signed_gap);
    features[17] = raw - previous_raw;
    features[18] = mapped - previous_mapped;
    features[19] = imu - previous_imu;
    features[20] = window_raw_speed;
    features[21] = body_speed_from_wheel_speed(window_raw_speed);
    // Keep source compatibility with the existing 22-feature production
    // header. A regenerated 24-feature candidate consumes these fields;
    // the old header simply leaves them out until that candidate is accepted.
    if constexpr (f1tenth_localization::SensorFusionModel::kFeatureCount > 22) {
      features[22] = imu_acceleration_bias;
    }
    if constexpr (f1tenth_localization::SensorFusionModel::kFeatureCount > 23) {
      features[23] = imu_observer_acceleration;
    }
    return features;
  }

  SensorMotionRegime classify_motion_regime() const
  {
    // Enter a new regime only after the larger threshold is crossed, but keep
    // the current accelerating/decelerating regime until acceleration has
    // returned through the smaller exit threshold. This prevents 40 Hz noise
    // from alternating the measurement variance at every sample.
    const double acceleration = imu_acceleration_filtered_mps2_;
    if (motion_regime_ == SensorMotionRegime::ACCELERATING &&
      acceleration > regime_exit_acceleration_mps2_)
    {
      return SensorMotionRegime::ACCELERATING;
    }
    if (motion_regime_ == SensorMotionRegime::DECELERATING &&
      acceleration < -regime_exit_acceleration_mps2_)
    {
      return SensorMotionRegime::DECELERATING;
    }
    if (acceleration > regime_enter_acceleration_mps2_) {
      return SensorMotionRegime::ACCELERATING;
    }
    if (acceleration < -regime_enter_acceleration_mps2_) {
      return SensorMotionRegime::DECELERATING;
    }
    return SensorMotionRegime::STEADY;
  }

  double motion_regime_code() const
  {
    switch (motion_regime_) {
      case SensorMotionRegime::ACCELERATING:
        return 0.0;
      case SensorMotionRegime::STEADY:
        return 1.0;
      case SensorMotionRegime::DECELERATING:
        return 2.0;
      case SensorMotionRegime::FROZEN:
        return 3.0;
      case SensorMotionRegime::FROZEN_ACCELERATING:
        return 3.0;
    }
    return 1.0;
  }

  void remember_sensor_fusion_sample(
    double raw, double mapped, double imu, double gap)
  {
    sensor_fusion_raw_history_.push_back(raw);
    sensor_fusion_mapped_history_.push_back(mapped);
    sensor_fusion_imu_history_.push_back(imu);
    sensor_fusion_gap_history_.push_back(gap);
    while (sensor_fusion_raw_history_.size() > 9) {
      sensor_fusion_raw_history_.pop_front();
      sensor_fusion_mapped_history_.pop_front();
      sensor_fusion_imu_history_.pop_front();
      sensor_fusion_gap_history_.pop_front();
    }
  }

  double sensor_fusion_window_raw_speed(
    double stamp_s, double position_m, double current_speed) const
  {
    if (sensor_fusion_encoder_history_.empty() ||
      stamp_s <= sensor_fusion_encoder_history_.front().first)
    {
      return current_speed;
    }
    // Encoder events can be delivered in zero/large bursts even though their
    // source timestamps are regular.  Select the most recent history sample
    // at or before a speed-dependent averaging horizon.  Low-speed samples
    // need the longer horizon to overcome angle quantisation; high-speed
    // samples use less history so acceleration response remains causal.
    const double previous_window_speed =
      std::max(0.0, sensor_fusion_window_raw_speed_mps_);
    const double reference_speed = std::max(
      std::abs(current_speed), previous_window_speed);
    double desired_duration = sensor_fusion_window_high_speed_duration_s_;
    if (reference_speed <= sensor_fusion_window_low_speed_threshold_mps_) {
      desired_duration = sensor_fusion_window_low_speed_duration_s_;
    } else if (reference_speed < sensor_fusion_window_high_speed_threshold_mps_) {
      desired_duration = sensor_fusion_window_mid_speed_duration_s_;
    }

    const double minimum_duration = std::min(0.10, desired_duration * 0.25);
    const double maximum_duration = std::min(
      1.00, std::max(0.30, desired_duration * 1.50));
    std::vector<std::pair<double, double>> samples(
      sensor_fusion_encoder_history_.begin(), sensor_fusion_encoder_history_.end());
    samples.emplace_back(stamp_s, position_m);
    std::vector<double> slopes;
    slopes.reserve(samples.size() * samples.size() / 2);
    for (size_t first_index = 0; first_index + 1 < samples.size(); ++first_index) {
      for (size_t after_index = first_index + 1;
        after_index < samples.size(); ++after_index)
      {
        const double duration = samples[after_index].first -
          samples[first_index].first;
        if (duration < minimum_duration || duration > maximum_duration) {
          continue;
        }
        const double slope = std::abs(
          samples[after_index].second - samples[first_index].second) / duration;
        if (std::isfinite(slope) && slope <= 40.0) {
          slopes.push_back(slope);
        }
      }
    }
    if (slopes.size() >= 4) {
      std::sort(slopes.begin(), slopes.end());
      return slopes[slopes.size() / 2];
    }

    // During the first few samples after a reset there is not yet enough
    // history for a robust slope. Fall back to the most recent valid pair.
    const auto reference = sensor_fusion_encoder_history_.empty() ?
      std::pair<double, double>{stamp_s, position_m} :
      sensor_fusion_encoder_history_.back();
    const double duration = stamp_s - reference.first;
    if (duration <= 1.0e-4) {
      return current_speed;
    }
    return std::abs(position_m - reference.second) / duration;
  }

  void remember_sensor_fusion_encoder_sample(double stamp_s, double position_m)
  {
    sensor_fusion_encoder_history_.emplace_back(stamp_s, position_m);
    // 45 samples cover the one-second low-speed horizon at the measured
    // 40 Hz source cadence, with room for timestamp jitter.
    while (sensor_fusion_encoder_history_.size() > 45) {
      sensor_fusion_encoder_history_.pop_front();
    }
  }

  void publish_odom(const rclcpp::Time & stamp, double speed)
  {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = odom_frame_;
    msg.child_frame_id = base_frame_;
    msg.pose.pose.position.x = x_;
    msg.pose.pose.position.y = y_;
    msg.pose.pose.orientation = quaternion_from_yaw(odom_yaw_);
    msg.twist.twist.linear.x = speed;
    msg.twist.twist.angular.z = imu_yaw_rate_;

    const double confidence = std::clamp(wheel_observation_confidence_, 0.0, 1.0);
    const double pose_variance = pose_xy_var_ +
      (slip_pose_xy_var_ - pose_xy_var_) * (1.0 - confidence);
    const double reset_age_s = encoder_reset_active_ ?
      (stamp - last_encoder_reset_stamp_).seconds() :
      std::numeric_limits<double>::infinity();
    const bool reset_covariance_active = encoder_reset_active_ &&
      reset_age_s >= 0.0 && reset_age_s <= encoder_reset_covariance_duration_s_;
    const double published_pose_variance = reset_covariance_active ?
      std::max(pose_variance, encoder_reset_pose_xy_var_) : pose_variance;
    const double published_twist_variance = reset_covariance_active ?
      std::max(twist_linear_var_, encoder_reset_twist_linear_var_) :
      twist_linear_var_;
    msg.pose.covariance[0] = published_pose_variance;
    msg.pose.covariance[7] = published_pose_variance;
    msg.pose.covariance[35] = pose_yaw_var_;
    msg.twist.covariance[0] = published_twist_variance;
    msg.twist.covariance[35] = twist_yaw_var_;

    std_msgs::msg::Float64MultiArray diagnostics;
    diagnostics.layout.dim.resize(1);
    diagnostics.layout.dim[0].label =
      "raw_wheel_speed_mps,corrected_wheel_speed_mps,longitudinal_slip_ratio,"
      "wheel_observation_confidence,imu_acceleration_bias_mps2,encoder_reset_count,"
      "imu_speed_mps,frozen_encoder_model_active,frozen_encoder_model_decel_mps2,"
      "sensor_fusion_model_speed_mps,sensor_fusion_model_spread_mps,"
      "sensor_fusion_model_active,sensor_motion_regime,"
      "sensor_fusion_window_raw_speed_mps,sensor_fusion_window_mapped_speed_mps,"
      "odom_stamp_s,imu_observer_acceleration_mps2,imu_pair_speed_mps,"
      "imu_pair_lead_s,odom_speed_mps,v2_global_speed_mps,"
      "v2_base_speed_mps,v2_braking_speed_mps,v2_braking_blend,v2_global_residual,"
      "v2_braking_residual,v2_active";
    diagnostics.layout.dim[0].size = 27;
    diagnostics.layout.dim[0].stride = 27;
    // Reset boundaries intentionally invalidate the instantaneous wheel
    // quantities below.  Keep the diagnostic vector finite for downstream
    // timing/data validation; encoder_reset_count_ remains the authoritative
    // reset marker and no invalid value is fed into odometry or the model.
    const auto finite_or_zero = [](double value) {
        return std::isfinite(value) ? value : 0.0;
    };
    diagnostics.data = {
      finite_or_zero(raw_wheel_speed_mps_),
      finite_or_zero(corrected_wheel_speed_mps_),
      finite_or_zero(longitudinal_slip_),
      finite_or_zero(confidence), finite_or_zero(longitudinal_observer_.bias()),
      static_cast<double>(encoder_reset_count_),
      finite_or_zero(imu_speed_mps_), frozen_encoder_model_active_ ? 1.0 : 0.0,
      finite_or_zero(frozen_encoder_model_decel_mps2_),
      finite_or_zero(sensor_fusion_model_speed_mps_),
      finite_or_zero(sensor_fusion_model_spread_mps_),
      sensor_fusion_model_active_ ? 1.0 : 0.0,
      motion_regime_code(), finite_or_zero(sensor_fusion_window_raw_speed_mps_),
      finite_or_zero(sensor_fusion_window_mapped_speed_mps_),
      finite_or_zero(stamp.seconds()),
      finite_or_zero(imu_observer_acceleration_mps2_),
      finite_or_zero(imu_pair_speed_mps_), finite_or_zero(imu_pair_lead_s_),
      finite_or_zero(speed_mps_), finite_or_zero(sensor_fusion_v2_global_speed_mps_),
      finite_or_zero(sensor_fusion_v2_base_speed_mps_),
      finite_or_zero(sensor_fusion_v2_braking_speed_mps_),
      finite_or_zero(sensor_fusion_v2_braking_blend_),
      finite_or_zero(sensor_fusion_v2_global_residual_),
      finite_or_zero(sensor_fusion_v2_braking_residual_),
      sensor_fusion_v2_active_ ? 1.0 : 0.0};
    diagnostics_pub_->publish(diagnostics);

    // Publish diagnostics first. The recorder subscribes to both topics and
    // can therefore associate the following odom sample with the diagnostic
    // vector produced for the same encoder pair. The explicit stamp fields
    // above remain the authoritative check if DDS scheduling interleaves the
    // callbacks.
    odom_pub_->publish(msg);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = msg.header;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = x_;
    tf.transform.translation.y = y_;
    tf.transform.rotation = msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);

  }

  std::mutex mutex_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reset_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::string odom_frame_;
  std::string base_frame_;

  double wheel_radius_{0.0590};
  double encoder_scale_{1.0};
  double max_encoder_step_m_{4.0};
  double max_encoder_step_speed_mps_{35.0};
  double max_encoder_pair_skew_s_{0.05};
  double pose_xy_var_{0.01};
  double pose_yaw_var_{0.01};
  double velocity_filter_alpha_{0.70};
  double velocity_filter_decel_alpha_{0.70};
  size_t velocity_median_window_{5};
  double max_velocity_accel_mps2_{40.0};
  double twist_linear_var_{0.04};
  double twist_yaw_var_{0.04};
  double encoder_reset_covariance_duration_s_{1.0};
  double encoder_reset_pose_xy_var_{0.25};
  double encoder_reset_twist_linear_var_{0.50};

  bool have_left_{false};
  bool have_right_{false};
  bool left_updated_{false};
  bool right_updated_{false};
  bool encoder_initialized_{false};
  bool imu_initialized_{false};

  double left_angle_{0.0};
  double right_angle_{0.0};
  double prev_left_{0.0};
  double prev_right_{0.0};
  rclcpp::Time left_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time right_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time prev_stamp_{0, 0, RCL_ROS_TIME};

  double imu_yaw_zero_{0.0};
  double imu_raw_yaw_{0.0};
  double last_imu_relative_yaw_{0.0};
  double imu_integrated_yaw_{0.0};
  double imu_yaw_rate_{0.0};
  double odom_yaw_{0.0};
  double prev_yaw_{0.0};
  double speed_mps_{0.0};
  std::deque<double> recent_raw_speeds_;
  rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};
  bool have_imu_stamp_{false};
  double imu_orientation_correction_gain_{0.08};
  double max_imu_orientation_step_rad_{0.30};
  double max_imu_dt_s_{0.5};
  double imu_pair_extrapolation_max_s_{0.10};
  double imu_acceleration_filter_alpha_{0.90};
  size_t imu_acceleration_median_window_{3};
  double wheel_slip_threshold_mps_{0.75};
  double wheel_slip_ratio_{0.20};
  double slip_observation_min_speed_mps_{0.75};
  double wheel_slip_activation_speed_mps_{16.0};
  double encoder_reset_threshold_rad_{0.50};
  double max_observer_accel_mps2_{12.0};
  double observer_speed_tolerance_mps_{2.0};
  double wheel_observer_correction_gain_{0.10};
  double observer_acceleration_noise_mps2_{1.50};
  double observer_bias_random_walk_mps3_{0.08};
  double observer_initial_speed_variance_m2ps2_{1.00};
  double observer_initial_bias_variance_m4ps4_{0.25};
  double observer_innovation_gate_mps_{1.50};
  double observer_encoder_measurement_variance_m2ps2_{0.09};
  double observer_accelerating_measurement_variance_m2ps2_{0.49};
  double observer_steady_measurement_variance_m2ps2_{0.04};
  double observer_decelerating_measurement_variance_m2ps2_{0.25};
  double imu_speed_correction_gain_{0.0};
  bool sensor_fusion_model_enabled_{true};
  bool sensor_fusion_v2_enabled_{true};
  double sensor_fusion_v2_braking_blend_offset_mps2_{0.35};
  double sensor_fusion_v2_braking_blend_range_mps2_{0.50};
  bool sensor_fusion_steady_model_enabled_{false};
  double regime_acceleration_threshold_mps2_{0.50};
  double regime_enter_acceleration_mps2_{0.65};
  double regime_exit_acceleration_mps2_{0.25};
  double regime_min_dwell_s_{0.075};
  double sensor_fusion_window_low_speed_duration_s_{1.00};
  double sensor_fusion_window_mid_speed_duration_s_{0.60};
  double sensor_fusion_window_high_speed_duration_s_{0.30};
  double sensor_fusion_window_low_speed_threshold_mps_{5.0};
  double sensor_fusion_window_high_speed_threshold_mps_{15.0};
  double sensor_fusion_window_stability_mps_{0.35};
  double sensor_fusion_wheel_downshift_min_drop_mps_{0.35};
  double sensor_fusion_wheel_downshift_hold_s_{0.50};
  double sensor_fusion_window_recovery_s_{0.35};
  double sensor_fusion_recovery_stability_mps_{0.50};
  double sensor_fusion_recovery_encoder_blend_{0.75};
  double sensor_fusion_recovery_encoder_max_correction_mps_{0.35};
  double sensor_fusion_model_blend_{0.50};
  double sensor_fusion_transient_model_blend_{0.50};
  double sensor_fusion_transient_model_max_correction_mps_{0.35};
  double sensor_fusion_transient_low_speed_threshold_mps_{0.75};
  double sensor_fusion_transient_mid_speed_threshold_mps_{1.50};
  double sensor_fusion_transient_low_speed_max_correction_mps_{0.10};
  double sensor_fusion_transient_mid_speed_max_correction_mps_{0.20};
  double low_speed_encoder_priority_mps_{5.0};
  double low_speed_encoder_max_imu_gap_mps_{0.75};
  double low_speed_encoder_stability_mps_{0.35};
  std::size_t low_speed_encoder_stability_samples_{8};
  double regime_model_max_spread_mps_{0.75};
  double regime_model_max_baseline_delta_mps_{0.75};
  bool steady_encoder_reanchor_enabled_{true};
  f1tenth_localization::SensorFusionModel sensor_fusion_model_;
  f1tenth_localization::LongitudinalObserver longitudinal_observer_;
  double stationary_speed_threshold_mps_{0.15};
  double forward_extremum_slip_{0.15};
  double forward_extremum_value_{0.72};
  double forward_asymptote_slip_{0.25};
  double forward_asymptote_value_{0.464};
  double imu_stop_speed_threshold_mps_{2.0};
  double imu_stationary_acceleration_threshold_mps2_{0.30};
  double zero_encoder_stop_confirm_sec_{0.80};
  bool frozen_encoder_model_enabled_{true};
  double frozen_encoder_model_delay_sec_{0.15};
  double frozen_encoder_decel_intercept_mps2_{5.5};
  double frozen_encoder_decel_speed_gain_per_s_{0.27};
  double frozen_encoder_decel_max_mps2_{12.0};
  double frozen_encoder_braking_model_blend_{0.75};
  double slip_pose_xy_var_{0.10};
  std::vector<double> wheel_speed_map_wheel_mps_;
  std::vector<double> wheel_speed_map_body_mps_;
  double imu_speed_mps_{0.0};
  double imu_acceleration_filtered_mps2_{0.0};
  double imu_acceleration_robust_mps2_{0.0};
  double imu_observer_acceleration_mps2_{0.0};
  std::deque<double> imu_acceleration_history_;
  double imu_raw_acceleration_mps2_{0.0};
  double imu_lateral_acceleration_mps2_{0.0};
  rclcpp::Time last_imu_speed_stamp_{0, 0, RCL_ROS_TIME};
  bool have_imu_speed_stamp_{false};
  bool imu_speed_ready_{false};
  bool wheel_slip_detected_{false};
  double wheel_observation_confidence_{1.0};
  bool frozen_encoder_model_active_{false};
  double frozen_encoder_model_decel_mps2_{0.0};
  double sensor_fusion_window_raw_speed_mps_{0.0};
  double sensor_fusion_window_mapped_speed_mps_{0.0};
  double raw_wheel_speed_mps_{0.0};
  double corrected_wheel_speed_mps_{0.0};
  double longitudinal_slip_{0.0};
  uint32_t encoder_reset_count_{0};
  bool encoder_reset_active_{false};
  rclcpp::Time last_encoder_reset_stamp_{0, 0, RCL_ROS_TIME};
  double zero_encoder_duration_s_{0.0};
  double last_motion_sign_{1.0};
  bool sensor_fusion_model_active_{false};
  double sensor_fusion_model_speed_mps_{0.0};
  double sensor_fusion_model_spread_mps_{0.0};
  double sensor_fusion_v2_global_speed_mps_{0.0};
  double sensor_fusion_v2_base_speed_mps_{0.0};
  double sensor_fusion_v2_braking_speed_mps_{0.0};
  double sensor_fusion_v2_braking_blend_{0.0};
  double sensor_fusion_v2_global_residual_{0.0};
  double sensor_fusion_v2_braking_residual_{0.0};
  bool sensor_fusion_v2_active_{false};
  double imu_pair_speed_mps_{0.0};
  double imu_pair_lead_s_{0.0};
  SensorMotionRegime motion_regime_{SensorMotionRegime::STEADY};
  SensorMotionRegime regime_candidate_{SensorMotionRegime::STEADY};
  double regime_candidate_dwell_s_{0.0};
  std::deque<double> sensor_fusion_raw_history_;
  std::deque<double> sensor_fusion_mapped_history_;
  std::deque<double> sensor_fusion_imu_history_;
  std::deque<double> sensor_fusion_gap_history_;
  std::deque<double> steady_mapped_speed_history_;
  std::deque<std::pair<double, double>> sensor_fusion_encoder_history_;
  std::deque<double> sensor_fusion_recovery_window_history_;
  f1tenth_localization::OdomEstimatorV2Model sensor_fusion_v2_model_;
  f1tenth_localization::OdomEstimatorV2Features sensor_fusion_v2_features_;
  bool sensor_fusion_braking_window_active_{false};
  double sensor_fusion_window_recovery_elapsed_s_{0.0};
  double sensor_fusion_wheel_downshift_hold_remaining_s_{0.0};
  bool sensor_fusion_wheel_reacquisition_required_{false};
  double x_{0.0};
  double y_{0.0};

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorOdometryNode>());
  rclcpp::shutdown();
  return 0;
}

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "f1tenth_localization/odometry_observer.hpp"
#include "f1tenth_localization/wheel_speed_map.hpp"

namespace
{

using f1tenth_localization::OdometryObservation;
using f1tenth_localization::OdometryObserver;
using f1tenth_localization::OdometryObserverConfig;

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

OdometryObservation observation(
  double stamp, double left, double right, double ax = 0.0,
  double ay = 0.0, double yaw_rate = 0.0, double yaw = 0.0)
{
  return {stamp, left, right, ax, ay, yaw_rate, yaw};
}

void test_stationary_and_wheel_gate()
{
  // This synthetic test deliberately uses 80 m/s^2 to seed a 2 m/s predicted
  // speed; the production default rejects such a longitudinal IMU value.
  OdometryObserverConfig synthetic_config;
  synthetic_config.max_imu_ax_abs_mps2 = 100.0;
  OdometryObserver observer(synthetic_config);
  auto first = observer.update(observation(0.0, 0.0, 0.0));
  auto second = observer.update(observation(0.025, 0.0, 0.0));
  require(first.valid && second.valid, "stationary packets are valid");
  require(second.speed_mps == 0.0, "stationary packets stay at zero");

  // A real low-speed packet must not be classified as stopped merely because
  // it is below the frozen-encoder protection threshold.
  observer.reset();
  observer.update(observation(0.0, 0.0, 0.0));
  const double low_speed_delta = 0.10 * 0.050 / 0.059;
  auto low_speed = observer.update(observation(
    0.050, low_speed_delta, low_speed_delta));
  require(low_speed.wheel_update_used, "low-speed wheel motion is accepted");
  require(low_speed.speed_mps > 0.08,
    "low-speed motion is not cleared by stationary detection");

  observer.reset();
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 80.0));
  const double one_packet_at_two_mps = 2.0 * 0.025 / 0.059;
  auto accepted = observer.update(observation(
    0.050, one_packet_at_two_mps, one_packet_at_two_mps));
  require(accepted.wheel_update_used, "small wheel innovation is accepted");

  observer.reset();
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 80.0));
  auto rejected = observer.update(observation(0.050, 8.0, 8.0));
  require(!rejected.wheel_update_used, "large wheel innovation is rejected");
}

void test_braking_and_frozen_wheel()
{
  OdometryObserverConfig synthetic_config;
  synthetic_config.max_imu_ax_abs_mps2 = 300.0;
  OdometryObserver observer(synthetic_config);
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 200.0));
  const auto before = observer.update(observation(0.050, 0.0, 0.0));
  const auto brake = observer.update(observation(0.075, 0.0, 0.0, -4.0));
  require(brake.speed_pred_mps < before.speed_mps, "braking reduces predicted speed");
  require(brake.speed_pred_mps > before.speed_mps - 0.11, "braking affine correction is small");
  const auto frozen = observer.update(observation(0.100, 10.0, 10.0, 0.0));
  require(frozen.speed_mps > 0.5, "frozen wheel does not zero moving speed");

  // A persistent stopped signal must eventually clear the protected moving
  // estimate; otherwise a simulator stop/collision leaves /odom moving
  // forever and the speed controller can never re-arm.
  auto stopped = frozen;
  for (int i = 0; i < 12; ++i) {
    stopped = observer.update(observation(0.125 + 0.025 * i, 10.0, 10.0));
  }
  require(stopped.speed_mps == 0.0, "persistent stationary evidence clears stale speed");
}

void test_epoch_and_timing()
{
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  auto regression = observer.update(observation(-0.01, 1.0, 1.0));
  require(regression.timing_degraded, "time regression is rejected");
  auto epoch = observer.update(observation(0.025, 51.0, 0.0));
  require(epoch.reset_epoch, "large encoder jump resets epoch");
  require(epoch.speed_mps == 0.0, "epoch reset rebaselines speed");
  auto gap = observer.update(observation(0.40, 51.0, 0.0));
  require(gap.timing_degraded, "long gap is degraded");

  OdometryObserver track_observer;
  track_observer.update(observation(0.0, 0.0, 0.0));
  const auto native_track_packet = track_observer.update(
    observation(0.050, 0.0, 0.0));
  require(!native_track_packet.timing_degraded,
    "native 20 Hz track packet is not degraded");

  OdometryObserver gap_observer;
  gap_observer.update(observation(0.0, 0.0, 0.0));
  const double speed = 0.30;
  const double first_delta = speed * 0.050 / 0.059;
  gap_observer.update(observation(0.050, first_delta, first_delta));
  const double gap_delta = speed * 0.150 / 0.059;
  const auto integrated_gap = gap_observer.update(
    observation(0.200, first_delta + gap_delta, first_delta + gap_delta));
  require(integrated_gap.timing_degraded,
    "a short source gap remains visible as degraded timing");
  require(integrated_gap.x_m > 0.05,
    "encoder endpoint displacement is retained across a short source gap");
}

void test_turn_mode_and_pose()
{
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  const double low_speed_delta = 0.5 * 0.050 / 0.059;
  auto entered = observer.update(observation(
    0.050, low_speed_delta, low_speed_delta, 0.0, 0.0, 0.7));
  require(entered.turn_mode, "lateral acceleration enters turn mode");
  require(entered.wheel_update_used, "turn mode uses valid wheel speed");
  require(entered.body_u_mps > 0.45, "turn mode preserves low-speed longitudinal motion");

  observer.update(observation(0.100, 2.0 * low_speed_delta, 2.0 * low_speed_delta,
    0.0, 7.0, 0.7));
  const auto no_lateral_drift = observer.update(observation(
    0.150, 3.0 * low_speed_delta, 3.0 * low_speed_delta, 0.0, 7.0, 0.7));
  require(no_lateral_drift.body_u_mps > 0.45,
    "turn mode remains anchored to the wheel speed");
  const auto lateral_accel_default = observer.update(observation(
    0.200, 4.0 * low_speed_delta, 4.0 * low_speed_delta, 0.0, 15.0, 0.7));
  require(std::abs(lateral_accel_default.body_v_mps) < 1.0e-12,
    "default turn model does not integrate unvalidated lateral acceleration");

  OdometryObserverConfig dynamic_config;
  dynamic_config.integrate_lateral_acceleration_in_turn = true;
  OdometryObserver dynamic_observer(dynamic_config);
  dynamic_observer.update(observation(0.0, 0.0, 0.0));
  dynamic_observer.update(observation(
    0.050, low_speed_delta, low_speed_delta, 0.0, 0.0, 0.7));
  const auto dynamic_turn = dynamic_observer.update(observation(
    0.100, 2.0 * low_speed_delta, 2.0 * low_speed_delta, 0.0, 7.0, 0.7));
  require(std::abs(dynamic_turn.body_v_mps) > 1.0e-5,
    "validated dynamic turn option still integrates lateral acceleration");

  auto calm = entered;
  for (int i = 0; i < 25; ++i) {
    calm = observer.update(observation(0.200 + 0.025 * i, 10.0, 10.0));
  }
  require(!calm.turn_mode, "turn mode exits only after calm hold");
  require(std::isfinite(calm.x_m) && std::isfinite(calm.y_m), "turn pose remains finite");
}

void test_impossible_longitudinal_acceleration_is_rejected()
{
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 0.0, 7.0));
  const auto before = observer.update(observation(0.050, 0.0, 0.0, 0.0, 7.0));
  const auto outlier = observer.update(
    observation(0.100, 0.0, 0.0, -122.75, 0.0));
  require(outlier.sensor_outlier, "impossible longitudinal acceleration is flagged");
  require(outlier.timing_degraded, "outlier is marked degraded");
  require(std::abs(outlier.x_m - before.x_m) < 1.0e-12,
    "outlier does not move odometry");
  require(outlier.speed_mps == before.speed_mps,
    "outlier holds the last causal speed");
  const auto recovered = observer.update(observation(0.150, 0.0, 0.0));
  require(!recovered.sensor_outlier, "observer recovers after one outlier");
  require(std::isfinite(recovered.x_m), "recovered pose remains finite");
}

void test_lever_arm_and_rk2_reference()
{
  OdometryObserverConfig config;
  config.imu_x_offset_m = 0.08;
  config.max_imu_ax_abs_mps2 = 300.0;
  OdometryObserver observer(config);
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 200.0));
  const auto result = observer.update(observation(
    0.050, 10.0, 10.0, 0.0, 0.0, 1.0));
  require(result.turn_mode, "yaw rate enters turn mode");
  require(std::isfinite(result.speed_mps), "RK2 turn result is finite");
  require(std::abs(f1tenth_localization::WheelSpeedMap::map(0.1) - 0.1) < 1.0e-12,
    "wheel map uses raw speed below calibration range");
  require(f1tenth_localization::WheelSpeedMap::map(30.0) > 20.0,
    "wheel map clamps to calibrated high-speed output");
}

}  // namespace

int main()
{
  test_stationary_and_wheel_gate();
  test_braking_and_frozen_wheel();
  test_epoch_and_timing();
  test_turn_mode_and_pose();
  test_impossible_longitudinal_acceleration_is_rejected();
  test_lever_arm_and_rk2_reference();
  std::cout << "odometry_observer_test: PASS" << std::endl;
  return EXIT_SUCCESS;
}

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
  OdometryObserver observer;
  auto first = observer.update(observation(0.0, 0.0, 0.0));
  auto second = observer.update(observation(0.025, 0.0, 0.0));
  require(first.valid && second.valid, "stationary packets are valid");
  require(second.speed_mps == 0.0, "stationary packets stay at zero");

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
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 200.0));
  const auto before = observer.update(observation(0.050, 0.0, 0.0));
  const auto brake = observer.update(observation(0.075, 0.0, 0.0, -4.0));
  require(brake.speed_pred_mps < before.speed_mps, "braking reduces predicted speed");
  require(brake.speed_pred_mps > before.speed_mps - 0.11, "braking affine correction is small");
  const auto frozen = observer.update(observation(0.100, 10.0, 10.0, 0.0));
  require(frozen.speed_mps > 0.5, "frozen wheel does not zero moving speed");
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
  auto gap = observer.update(observation(0.20, 51.0, 0.0));
  require(gap.timing_degraded, "long gap is degraded");
}

void test_turn_mode_and_pose()
{
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 200.0));
  auto entered = observer.update(observation(0.050, 0.0, 0.0, 0.0, 7.0));
  require(entered.turn_mode, "lateral acceleration enters turn mode");
  require(std::abs(entered.body_v_mps) > 0.0, "turn mode propagates lateral velocity");

  auto calm = entered;
  for (int i = 0; i < 25; ++i) {
    calm = observer.update(observation(0.075 + 0.025 * i, 10.0, 10.0));
  }
  require(!calm.turn_mode, "turn mode exits only after calm hold");
  require(std::isfinite(calm.x_m) && std::isfinite(calm.y_m), "turn pose remains finite");
}

void test_lever_arm_and_rk2_reference()
{
  OdometryObserverConfig config;
  config.imu_x_offset_m = 0.08;
  OdometryObserver observer(config);
  observer.update(observation(0.0, 0.0, 0.0));
  observer.update(observation(0.025, 0.0, 0.0, 200.0));
  const auto result = observer.update(observation(
    0.050, 10.0, 10.0, 0.0, 0.0, 1.0));
  require(result.turn_mode, "yaw rate enters turn mode");
  require(std::isfinite(result.speed_mps), "RK2 turn result is finite");
  require(std::abs(f1tenth_localization::WheelSpeedMap::map(0.1)) < 1.0e-12,
    "wheel map returns zero below calibration range");
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
  test_lever_arm_and_rk2_reference();
  std::cout << "odometry_observer_test: PASS" << std::endl;
  return EXIT_SUCCESS;
}

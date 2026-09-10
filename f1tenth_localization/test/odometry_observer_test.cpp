#include <cmath>
#include <cstdlib>
#include <iostream>

#include "f1tenth_localization/odometry_observer.hpp"

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
  const double one_packet_at_two_mps = 2.0 * 0.050 / 0.059;
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

void test_launch_uses_wheel_speed_during_acceleration()
{
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  const double launch_speed = 1.8;
  const double launch_delta = launch_speed * 0.050 / 0.059;
  const auto launch = observer.update(observation(
    0.050, launch_delta, launch_delta, 2.0));
  require(launch.wheel_update_used,
    "valid wheel speed is used during normal launch acceleration");
  require(launch.speed_mps > 1.2,
    "launch odometry follows the wheel measurement instead of IMU lag");
}

void test_pose_integration_uses_velocity_midpoint()
{
  OdometryObserverConfig config;
  config.wheel_speed_scale = 1.0;
  OdometryObserver observer(config);
  observer.update(observation(0.0, 0.0, 0.0));

  // The first wheel sample establishes 2 m/s at t=0.05 s. A causal pose
  // integrator must use the midpoint velocity for that interval: 0.05 m,
  // not the endpoint velocity over the whole interval (0.10 m).
  const double delta = 2.0 * 0.050 / config.wheel_radius_m;
  const auto first = observer.update(observation(0.050, delta, delta));
  require(std::abs(first.x_m - 0.050) < 1.0e-9,
    "launch displacement uses the velocity midpoint");

  const auto second = observer.update(observation(0.100, 2.0 * delta, 2.0 * delta));
  require(std::abs(second.x_m - 0.150) < 1.0e-9,
    "constant-speed displacement remains exact after launch");
}

void test_turn_entry_uses_coherent_launch_wheel_speed()
{
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  const double launch_speed = 1.8;
  const double launch_delta = launch_speed * 0.050 / 0.059;
  observer.update(observation(0.050, launch_delta, launch_delta, 1.0, 0.0, 0.0));

  // A steering transient enters turn mode while the IMU prediction is still
  // below the wheel speed. The coherent wheel packet must seed the turn.
  const auto entered = observer.update(observation(
    0.100, 2.0 * launch_delta, 2.0 * launch_delta, 1.0, 0.0, 0.7));
  require(entered.turn_mode, "launch steering transient enters turn mode");
  require(entered.wheel_update_used, "turn entry accepts coherent wheel speed");
  require(entered.body_u_mps > 1.5,
    "turn entry does not retain the stale low IMU launch prediction");
}

void test_turn_braking_recovers_nonzero_wheel_sample()
{
  OdometryObserverConfig synthetic_config;
  synthetic_config.reset_encoder_jump_rad = 10.0;
  OdometryObserver observer(synthetic_config);
  observer.update(observation(0.0, 0.0, 0.0));
  const double fast_delta = 4.0 * 0.050 / 0.059;
  observer.update(observation(0.050, fast_delta, fast_delta));
  const auto fast = observer.update(observation(0.100, 2.0 * fast_delta, 2.0 * fast_delta));
  require(fast.body_u_mps > 3.0, "test seeds a moving over-high estimate");

  // Enter a turn while the synchronized wheel packet says the car has
  // already slowed.  The normal innovation gate would reject this packet;
  // the braking-specific path must accept it.
  const double recovery_delta = 1.2 * 0.050 / 0.059;
  const auto recovered = observer.update(observation(
    0.150, 2.0 * fast_delta + recovery_delta,
    2.0 * fast_delta + recovery_delta, -2.0, 0.0, 0.7));
  require(recovered.wheel_update_used,
    "nonzero wheel packet is accepted during hard turn braking");
  require(recovered.body_u_mps < fast.body_u_mps,
    "braking wheel recovery removes stale over-high speed");

  const auto frozen = observer.update(observation(
    0.200, 2.0 * fast_delta + recovery_delta,
    2.0 * fast_delta + recovery_delta, -2.0, 0.0, 0.7));
  require(frozen.body_u_mps > 0.5,
    "isolated zero wheel packet remains protected during braking");
}

void test_repeated_encoder_packet_uses_imu_until_recovery()
{
  OdometryObserver observer;
  observer.update(observation(0.0, 0.0, 0.0));
  const double fast_delta = 4.0 * 0.050 / 0.059;
  observer.update(observation(0.050, fast_delta, fast_delta, 0.0, 0.0, 0.7));
  const auto before = observer.update(observation(
    0.100, 2.0 * fast_delta, 2.0 * fast_delta, 0.0, 0.0, 0.7));
  require(before.body_u_mps > 3.0, "test seeds a fast turn speed");

  // This is the packet pattern observed in the clean track runs: the
  // cumulative encoder angle repeats while the car is still braking.
  const auto missing = observer.update(observation(
    0.150, 2.0 * fast_delta, 2.0 * fast_delta, -6.2, 0.0, 0.7));
  require(!missing.wheel_update_used,
    "repeated encoder angle is not used as a zero-speed wheel update");
  require(missing.body_u_mps > 3.4,
    "IMU braking propagation preserves motion through missing encoder data");

  const double recovery_delta = 3.4 * 0.050 / 0.059;
  const auto recovered = observer.update(observation(
    0.200, 2.0 * fast_delta + recovery_delta,
    2.0 * fast_delta + recovery_delta, 0.0, 0.0, 0.7));
  require(recovered.wheel_update_used,
    "coherent encoder displacement re-anchors after the dropout");
  require(recovered.body_u_mps > 3.0,
    "dropout recovery does not reintroduce the lagging window speed");
}

void test_near_freeze_encoder_packet_cannot_collapse_motion()
{
  OdometryObserverConfig config;
  config.reset_encoder_jump_rad = 10.0;
  OdometryObserver observer(config);
  observer.update(observation(0.0, 0.0, 0.0));
  const double fast_delta = 4.0 * 0.050 / 0.059;
  observer.update(observation(0.050, fast_delta, fast_delta, 0.0, 0.0, 0.7));
  const auto before = observer.update(observation(
    0.100, 2.0 * fast_delta, 2.0 * fast_delta, 0.0, 0.0, 0.7));
  require(before.body_u_mps > 3.0, "test seeds a moving turn speed");

  // This is deliberately just above wheel_freeze_speed_mps.  It represents
  // a repeated cumulative encoder angle that used to pass the coherent-rate
  // shortcut because both the rolling and packet rates were equally stale.
  const double near_freeze_delta = 0.16 * 0.050 / 0.059;
  const auto missing = observer.update(observation(
    0.150, 2.0 * fast_delta + near_freeze_delta,
    2.0 * fast_delta + near_freeze_delta, -6.0, 0.0, 0.7));
  require(!missing.wheel_update_used,
    "near-freeze repeated encoder packet is rejected");
  require(missing.body_u_mps > 3.0,
    "near-freeze packet does not collapse causal turn speed");
}

void test_recovery_rebases_stale_window()
{
  OdometryObserverConfig config;
  config.reset_encoder_jump_rad = 10.0;
  OdometryObserver observer(config);
  observer.update(observation(0.0, 0.0, 0.0));
  const double fast_delta = 4.0 * 0.050 / 0.059;
  observer.update(observation(0.050, fast_delta, fast_delta, 0.0, 0.0, 0.7));
  observer.update(observation(0.100, 2.0 * fast_delta, 2.0 * fast_delta,
    0.0, 0.0, 0.7));

  // Repeated cumulative angles mark a missing packet while the car is still
  // moving. The following packet recovers motion and re-baselines the rolling
  // window, so the next normal update must not report a stale low speed.
  observer.update(observation(
    0.150, 2.0 * fast_delta, 2.0 * fast_delta, -6.2, 0.0, 0.7));
  const double recovery_delta = 3.4 * 0.050 / 0.059;
  observer.update(observation(
    0.200, 2.0 * fast_delta + recovery_delta,
    2.0 * fast_delta + recovery_delta, 0.0, 0.0, 0.7));
  const double current_delta = 4.0 * 0.050 / 0.059;
  const auto coherent = observer.update(observation(
    0.250, 2.0 * fast_delta + recovery_delta + current_delta,
    2.0 * fast_delta + recovery_delta + current_delta,
    0.0, 0.0, 0.7));
  require(coherent.wheel_update_used,
    "normal wheel update remains accepted after recovery");
  require(coherent.body_u_mps > 3.5,
    "rebased rolling window does not retain the stale low speed");
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

void test_encoder_burst_is_rejected_until_coherent_recovery()
{
  OdometryObserverConfig config;
  config.reset_encoder_jump_rad = 50.0;
  config.wheel_innovation_max_mps = 0.5;
  OdometryObserver observer(config);
  observer.update(observation(0.0, 0.0, 0.0));
  const double normal_delta = 4.0 * 0.050 / 0.059;
  observer.update(observation(0.050, normal_delta, normal_delta));
  const auto before = observer.update(observation(
    0.100, 2.0 * normal_delta, 2.0 * normal_delta));
  require(before.body_u_mps > 3.5, "test seeds the causal speed");

  // The rolling window sees a delayed jump, followed by an even larger
  // current-packet rate. This is the two-stage signature seen in run09.
  const double burst_delta = 9.0 * 0.050 / 0.059;
  const auto burst = observer.update(observation(
    0.150, 2.0 * normal_delta + burst_delta,
    2.0 * normal_delta + burst_delta));
  require(burst.wheel_burst_rejected,
    "two-stage cumulative encoder burst is diagnosed");
  require(!burst.wheel_update_used,
    "two-stage cumulative encoder burst is not used for speed");
  require(burst.body_u_mps < before.body_u_mps + 0.2,
    "burst does not inflate causal speed");

  const auto repeated = observer.update(observation(
    0.200, 2.0 * normal_delta + burst_delta,
    2.0 * normal_delta + burst_delta));
  require(!repeated.wheel_update_used,
    "repeated burst packet remains excluded");

  // The first nonzero packet after the burst can be the delayed low-rate
  // half of that same burst. It is close enough to the causal speed to pass
  // the innovation gate, but must not be accepted while it disagrees with
  // the rolling window.
  const double delayed_delta = 2.5 * 0.025 / 0.059;
  const auto delayed = observer.update(observation(
    0.225, 2.0 * normal_delta + burst_delta + delayed_delta,
    2.0 * normal_delta + burst_delta + delayed_delta));
  require(!delayed.wheel_update_used,
    "delayed low-rate burst packet remains excluded");
  require(delayed.speed_mps > before.speed_mps - 0.2,
    "delayed burst packet does not collapse causal speed");

  const auto stale = observer.update(observation(
    0.275, 2.0 * normal_delta + burst_delta + delayed_delta + 6.0 * 0.050 / 0.059,
    2.0 * normal_delta + burst_delta + delayed_delta + 6.0 * 0.050 / 0.059));
  require(!stale.wheel_update_used,
    "stale-window packet remains excluded before coherent recovery");

  // With the deployed 100 ms window, one additional coherent high-speed
  // packet is needed to age the delayed burst out of the rolling estimate.
  // It is intentionally far above the stale causal speed: agreement between
  // the current packet and rolling rate is the evidence that recovery is safe.
  const double coherent_delta = 6.0 * 0.050 / 0.059;
  const auto recovered = observer.update(observation(
    0.325, 2.0 * normal_delta + burst_delta + delayed_delta + 2.0 * coherent_delta,
    2.0 * normal_delta + burst_delta + delayed_delta + 2.0 * coherent_delta));
  require(recovered.wheel_update_used,
    "coherent packet clears the burst dropout");
  require(!recovered.wheel_burst_rejected,
    "coherent recovery packet is not diagnosed as a burst");
}

void test_coherent_wheel_rate_overrides_stale_innovation_gate()
{
  OdometryObserverConfig config;
  config.wheel_innovation_max_mps = 0.5;
  OdometryObserver observer(config);
  observer.update(observation(0.0, 0.0, 0.0));
  const double normal_delta = 4.0 * 0.050 / 0.059;
  const double fast_delta = 6.0 * 0.050 / 0.059;
  observer.update(observation(0.050, normal_delta, normal_delta));

  // The rolling window is 5 m/s and the current packet is 6 m/s.  They are
  // mutually coherent, but both exceed the stale 4 m/s causal estimate by
  // more than the ordinary innovation gate.
  const auto recovered = observer.update(observation(
    0.100, normal_delta + fast_delta, normal_delta + fast_delta));
  require(recovered.wheel_update_used,
    "coherent wheel rate overrides stale innovation gate");
  require(recovered.speed_mps > 4.5,
    "coherent wheel rate updates the stale speed estimate");
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
  require(std::abs(result.wheel_mapped_mps - 0.968 * result.wheel_raw_mps) < 1.0e-9,
    "wheel speed uses direct scale calibration without a stale lookup table");
}

}  // namespace

int main()
{
  test_stationary_and_wheel_gate();
  test_braking_and_frozen_wheel();
  test_epoch_and_timing();
  test_turn_mode_and_pose();
  test_launch_uses_wheel_speed_during_acceleration();
  test_pose_integration_uses_velocity_midpoint();
  test_turn_entry_uses_coherent_launch_wheel_speed();
  test_impossible_longitudinal_acceleration_is_rejected();
  test_lever_arm_and_rk2_reference();
  test_turn_braking_recovers_nonzero_wheel_sample();
  test_repeated_encoder_packet_uses_imu_until_recovery();
  test_near_freeze_encoder_packet_cannot_collapse_motion();
  test_recovery_rebases_stale_window();
  test_encoder_burst_is_rejected_until_coherent_recovery();
  test_coherent_wheel_rate_overrides_stale_innovation_gate();
  std::cout << "odometry_observer_test: PASS" << std::endl;
  return EXIT_SUCCESS;
}

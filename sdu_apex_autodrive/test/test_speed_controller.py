from dataclasses import replace

from sdu_apex_autodrive.speed_controller import (
    LongitudinalStateEstimator,
    SpeedControllerConfig,
    TargetAccelerationController,
    TargetSpeedController,
)


def config():
    return SpeedControllerConfig(
        kp=0.02,
        ki=0.01,
        ka=0.0,
        integral_limit=1.0,
        throttle_max_forward=0.10,
        throttle_rise_rate_per_sec=1.0,
        throttle_fall_rate_per_sec=2.0,
        stop_speed_threshold_mps=0.02,
        overspeed_coast_threshold_mps=0.08,
        feedforward_speed_mps=(0.0, 1.0, 2.0),
        feedforward_throttle=(0.0, 0.03, 0.06),
    )


def test_feedforward_interpolation():
    controller = TargetSpeedController(config())
    assert abs(controller.feedforward(1.5) - 0.045) < 1e-12


def test_stop_resets_controller():
    controller = TargetSpeedController(config())
    controller.update(1.0, 0.0, 0.0, 0.1)
    assert controller.update(0.0, 0.2, 0.0, 0.1) == 0.0
    assert controller.integral == 0.0


def test_output_never_exceeds_limit():
    controller = TargetSpeedController(config())
    for _ in range(100):
        output = controller.update(10.0, 0.0, 0.0, 0.1)
    assert 0.0 <= output <= 0.10


def test_material_overspeed_coasts_and_clears_integral():
    controller = TargetSpeedController(replace(
        config(), speed_hold_error_deadband_mps=0.0))
    # Keep the first demand below this deliberately small test actuator's
    # throttle ceiling so the integral has a chance to accumulate.
    controller.update(0.2, 0.0, 0.0, 0.5)
    assert controller.integral > 0.0

    output = controller.update(0.2, 0.7, 0.0, 0.1)
    assert output < 0.1
    assert controller.integral == 0.0


def test_10hz_rise_and_fall_are_slew_limited():
    controller = TargetSpeedController(config())

    # The calibration and AutoDRIVE command path are event-driven at 10 Hz.
    rising = [controller.update(5.5, 0.0, 0.0, 0.1) for _ in range(4)]
    assert rising == sorted(rising)
    assert rising[0] <= 0.10 + 1e-12
    assert all(b - a <= 0.10 + 1e-12 for a, b in zip(rising, rising[1:]))

    falling = [controller.update(0.5, 5.5, 0.0, 0.1) for _ in range(3)]
    assert all(a - b <= 0.20 + 1e-12 for a, b in zip(falling, falling[1:]))
    assert all(0.0 <= value <= 0.10 for value in falling)


def test_speed_controller_holds_calibrated_feedforward_when_settled():
    controller = TargetSpeedController(config())
    controller.update(2.0, 0.0, 0.0, 0.1)

    expected = controller.feedforward(2.0)
    held = [controller.update(2.0, 1.9, 0.0, 0.1) for _ in range(4)]

    assert held == [expected] * 4


def test_speed_controller_leaves_hold_deadband_for_fast_correction():
    controller = TargetSpeedController(config())
    controller.update(2.0, 1.9, 0.0, 0.1)
    expected = controller.feedforward(2.0)

    output = controller.update(2.0, 0.0, 0.0, 0.1)

    assert output > expected


def test_speed_controller_boosts_at_full_throttle_far_below_target():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
    ))

    output = controller.update(5.0, 0.0, 0.0, 0.1)

    assert output == 1.0


def test_speed_controller_handoffs_to_target_hold_throttle_before_crossing():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
    ))
    controller.update(5.0, 0.0, 0.0, 0.1)

    # The previous full-throttle command predicts that the target will be
    # crossed shortly, so the controller selects the mapped 5 m/s hold value.
    output = controller.update(5.0, 3.8, 0.0, 0.1)

    assert output == controller.feedforward(5.0)
    assert controller._hold_approach

    # Do not re-enter boost while the vehicle is still approaching the target.
    assert controller.update(5.0, 4.8, 0.0, 0.1) == controller.feedforward(5.0)


def test_speed_controller_keeps_hold_throttle_through_small_crossing():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
    ))
    controller.update(2.0, 1.9, 0.0, 0.1)
    expected = controller.feedforward(2.0)

    assert controller.update(2.0, 2.05, 0.0, 0.1) == expected


def test_speed_controller_exits_hold_for_material_underspeed():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
    ))
    controller.update(2.0, 1.9, 0.0, 0.1)
    output = controller.update(2.0, 1.5, 0.0, 0.1)

    assert not controller._hold_approach
    assert output > controller.feedforward(2.0)
    assert controller.update(2.0, 1.5, 0.0, 0.1) > controller.feedforward(2.0)


def test_speed_controller_can_tolerate_bounded_hold_under_report():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
        speed_hold_recovery_error_mps=1.0,
    ))
    controller.update(5.0, 0.0, 0.0, 0.1)
    expected = controller.feedforward(5.0)
    assert controller.update(5.0, 3.8, 0.0, 0.1) == expected
    assert controller.update(5.0, 4.0, 0.0, 0.1) == expected


def test_speed_controller_target_change_exits_hold_approach():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
    ))
    controller.update(5.0, 0.0, 0.0, 0.1)
    controller.update(5.0, 3.8, 0.0, 0.1)

    assert controller.update(10.0, 3.8, 0.0, 0.1) == 1.0


def test_speed_controller_downshift_waits_for_fresh_stable_odom():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
        speed_downshift_stable_sec=0.2,
        speed_downshift_band_mps=0.1,
    ))
    controller.update(5.0, 4.9, 0.0, 0.1)

    # A lower target must immediately coast while the old speed is still
    # above the new band.
    assert controller.update(2.0, 5.0, 0.0, 0.1) == 0.0
    assert controller._downshift_guard

    # A repeated snapshot is not fresh evidence and cannot restore throttle.
    assert controller.update(
        2.0, 2.05, 0.0, 0.1, measurement_fresh=False) == 0.0
    assert controller._downshift_guard

    # One fresh in-band sample is still insufficient; the second completes
    # the dwell and only then restores the calibrated hold throttle.
    assert controller.update(2.0, 2.05, 0.0, 0.1) == 0.0
    assert controller._downshift_guard
    assert controller.update(2.0, 2.05, 0.0, 0.1) == controller.feedforward(2.0)
    assert not controller._downshift_guard


def test_speed_controller_catches_predicted_downshift_coast():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
        speed_hold_prediction_horizon_sec=0.25,
        speed_downshift_stable_sec=0.2,
        speed_downshift_band_mps=0.1,
    ))
    controller.update(10.0, 9.9, 0.0, 0.1)

    # The car is still above the 2 m/s target, but the allowed IMU signal
    # predicts that passive coast would cross the target before the next
    # control update. Start the new target hold throttle immediately.
    output = controller.update(4.0, 5.0, 0.0, 0.1, measured_accel_mps2=-6.0)
    assert controller._downshift_catch
    assert output == controller.feedforward(4.0)

    # Once fresh odometry reaches the target band, the catch phase ends while
    # the calibrated hold throttle remains active.
    assert controller.update(4.0, 4.05, 0.0, 0.1) == controller.feedforward(4.0)
    assert not controller._downshift_catch


def test_speed_controller_uses_relative_overspeed_guard():
    controller = TargetSpeedController(replace(
        config(),
        throttle_max_forward=1.0,
        throttle_rise_rate_per_sec=10.0,
        throttle_fall_rate_per_sec=10.0,
        overspeed_coast_threshold_mps=1.0,
    ))

    # A 0.6 m/s overspeed is below the old fixed 1 m/s threshold but above
    # the new 10% target-relative guard for a 5 m/s target.
    assert controller.update(5.0, 5.6, 0.0, 0.1) == 0.0


def test_speed_loop_uses_speed_slew_limits_not_acceleration_limits():
    controller = TargetSpeedController(replace(
        config(),
        acceleration_throttle_rise_rate_per_sec=0.1,
        acceleration_throttle_fall_rate_per_sec=0.1,
    ))
    output = controller.update(5.5, 0.0, 0.0, 0.1)
    assert output == 0.10


def test_acceleration_loop_uses_its_independent_slew_limits():
    controller = TargetAccelerationController(replace(
        config(),
        acceleration_throttle_rise_rate_per_sec=0.1,
        acceleration_throttle_fall_rate_per_sec=0.1,
    ))
    output = controller.update(5.0, 0.0, 0.0, 0.1)
    assert abs(output - 0.01) < 1.0e-12


def test_feedforward_and_output_remain_bounded_above_measured_table():
    controller = TargetSpeedController(config())
    previous = 0.0
    for target in (0.0, 0.5, 1.0, 2.0, 3.5, 5.5):
        feedforward = controller.feedforward(target)
        output = controller.update(target, 0.0, 0.0, 0.1)
        assert 0.0 <= feedforward <= 0.10
        assert 0.0 <= output <= 0.10
        assert output >= previous
        previous = output


def test_acceleration_demand_uses_full_normalized_throttle_range():
    controller = TargetSpeedController(config())
    outputs = [controller.update(20.0, 0.0, 0.0, 0.1, 0.0) for _ in range(8)]
    assert max(outputs) == 0.10
    assert min(outputs) >= 0.0


def test_acceleration_controller_reduces_throttle_for_measured_overshoot():
    controller = TargetSpeedController(config())
    controller.update(2.0, 0.0, 0.0, 0.1, 0.0)
    output = controller.update(2.0, 1.0, 0.0, 0.1, 25.0)
    assert 0.0 <= output < 0.10


def test_throttle_to_acceleration_conversion_round_trips():
    controller = TargetSpeedController(config())
    acceleration = controller.acceleration_controller.acceleration_from_throttle(
        speed_mps=4.0, throttle=0.08, base_throttle=0.03)
    assert acceleration > 0.0
    gain = controller.acceleration_controller.throttle_per_acceleration(4.0)
    assert abs(acceleration * gain - 0.05) < 1.0e-12


def test_acceleration_to_throttle_conversion_uses_absolute_base_command():
    controller = TargetSpeedController(config())
    acceleration = 1.5
    command = controller.acceleration_controller.throttle_for_acceleration(
        speed_mps=4.0, base_throttle=0.03, desired_accel_mps2=acceleration)
    gain = controller.acceleration_controller.throttle_per_acceleration(4.0)
    assert abs(command - (0.03 + gain * acceleration)) < 1.0e-12
    recovered = controller.acceleration_controller.acceleration_from_throttle(
        speed_mps=4.0, throttle=command, base_throttle=0.03)
    assert abs(recovered - acceleration) < 1.0e-12


def test_conversion_is_defined_over_the_full_official_speed_domain():
    controller = TargetSpeedController(config())
    acceleration_controller = controller.acceleration_controller
    for speed in (0.0, 2.0, 5.5, 10.0, 15.0, 17.9, 20.0, 22.88):
        base = controller.feedforward(min(speed, 2.0))
        command = acceleration_controller.throttle_for_acceleration(
            speed, base, 1.0)
        recovered = acceleration_controller.acceleration_from_throttle(
            speed, command, base)
        assert abs(recovered - 1.0) < 1.0e-12


def test_calibrated_odom_remains_absolute_when_imu_disagrees():
    observer = LongitudinalStateEstimator(
        acceleration_filter_alpha=1.0,
        slip_threshold_mps=0.5,
        slip_ratio=0.2,
        odom_correction_gain=0.25,
    )
    observer.update_odometry(0.0, 0.0)
    observer.update_acceleration(4.0, 0.0)
    observer.update_acceleration(4.0, 0.1)
    estimate_before_spin = observer.speed_mps
    observer.update_odometry(10.0, 0.1)
    assert observer.slip_detected
    assert observer.speed_mps == 10.0
    assert observer.speed_mps != estimate_before_spin


def test_imu_observer_anchors_when_encoder_and_imu_agree():
    observer = LongitudinalStateEstimator(acceleration_filter_alpha=1.0)
    observer.update_odometry(2.0, 0.0)
    observer.update_acceleration(0.0, 0.0)
    observer.update_acceleration(0.0, 0.1)
    observer.update_odometry(2.4, 0.1)
    assert not observer.slip_detected
    assert observer.speed_mps == 2.4


def test_imu_update_does_not_overwrite_absolute_odom_speed():
    observer = LongitudinalStateEstimator(acceleration_filter_alpha=1.0)
    observer.update_odometry(5.0, 0.0)
    observer.update_acceleration(8.0, 0.0)
    observer.update_acceleration(8.0, 0.1)
    assert observer.speed_mps == 5.0
    assert observer.acceleration_mps2 == 8.0


def test_acceleration_command_uses_speed_only_for_feedforward():
    controller = TargetAccelerationController(config())
    output = controller.update(
        target_accel_mps2=0.0,
        measured_speed_mps=1.5,
        measured_accel_mps2=0.0,
        dt_seconds=0.1,
    )
    # The zero-acceleration command still holds the measured speed with the
    # feed-forward throttle; it is not interpreted as a stop request.
    assert 0.0 < output <= 0.10


def test_acceleration_command_coasts_for_negative_acceleration():
    controller = TargetAccelerationController(config())
    controller.update(1.0, 1.0, 0.0, 0.1)
    assert controller.update(-1.0, 1.0, 0.0, 0.1) == 0.0
    assert controller.last_output == 0.0


def test_acceleration_command_is_slew_limited_and_bounded():
    controller = TargetAccelerationController(config())
    outputs = [controller.update(5.0, 0.0, 0.0, 0.1) for _ in range(20)]
    assert outputs == sorted(outputs)
    assert all(0.0 <= value <= 0.10 for value in outputs)
    assert all(b - a <= 0.10 + 1.0e-12 for a, b in zip(outputs, outputs[1:]))


def test_acceleration_controller_exposes_speed_dependent_capability():
    controller = TargetAccelerationController(config())

    at_zero = controller.acceleration_controller.maximum_acceleration(0.0)
    at_ten = controller.acceleration_controller.maximum_acceleration(10.0)
    at_top = controller.acceleration_controller.maximum_acceleration(23.0)

    assert abs(at_zero - 0.63) < 0.01
    assert at_ten < at_zero
    assert at_top < at_ten

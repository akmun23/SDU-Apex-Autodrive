from sdu_apex_autodrive.speed_controller import (
    LongitudinalStateEstimator,
    SpeedControllerConfig,
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
    controller = TargetSpeedController(config())
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

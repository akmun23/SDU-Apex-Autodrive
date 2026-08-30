import pytest

from sdu_apex_autodrive.speed_controller import (
    SpeedControllerConfig,
    TargetSpeedController,
)


def config(**overrides):
    values = dict(
        kp=0.0,
        ki=0.0,
        ka=0.0,
        integral_limit=1.0,
        throttle_min_forward=0.0,
        throttle_max_forward=0.10,
        throttle_rise_rate_per_sec=100.0,
        throttle_fall_rate_per_sec=100.0,
        stop_speed_threshold_mps=0.02,
        feedforward_speed_mps=(0.0, 1.0, 2.0),
        feedforward_throttle=(0.0, 0.04, 0.07),
    )
    values.update(overrides)
    return SpeedControllerConfig(**values)


def update(controller, target=1.0, measured=1.0, accel=0.0, dt=0.1):
    return controller.update(target, measured, accel, dt)


def test_feedforward_only_output():
    assert update(TargetSpeedController(config())) == pytest.approx(0.04)


def test_signed_proportional_correction():
    controller = TargetSpeedController(config(kp=0.02))
    assert update(controller, measured=0.5) > 0.04
    controller.reset()
    assert update(controller, measured=1.5) < 0.04


def test_integral_accumulates_and_clamps():
    controller = TargetSpeedController(config(ki=0.01, integral_limit=0.2))
    for _ in range(10):
        update(controller, measured=0.0, dt=0.1)
    assert controller.integral == pytest.approx(0.2)


def test_anti_windup_at_positive_saturation():
    controller = TargetSpeedController(config(
        kp=1.0,
        ki=1.0,
        feedforward_speed_mps=(0.0, 1.0),
        feedforward_throttle=(0.0, 0.10),
    ))
    update(controller, target=1.0, measured=0.0)
    assert controller.integral == 0.0


def test_target_zero_resets_controller():
    controller = TargetSpeedController(config(ki=0.01))
    update(controller, measured=0.0)
    assert controller.integral > 0.0
    assert update(controller, target=0.0, measured=1.0) == 0.0
    assert controller.integral == 0.0


def test_independent_rise_and_fall_slew_limits():
    controller = TargetSpeedController(config(
        throttle_rise_rate_per_sec=0.10,
        throttle_fall_rate_per_sec=0.02,
    ))
    rising = update(controller, target=2.0, measured=0.0, dt=0.1)
    assert rising == pytest.approx(0.01)
    falling = update(controller, target=0.1, measured=2.0, dt=0.1)
    assert falling == pytest.approx(0.008)


def test_acceleration_feedforward():
    controller = TargetSpeedController(config(ka=0.01))
    assert update(controller, accel=2.0) == pytest.approx(0.06)


def test_output_never_exceeds_configured_limit():
    controller = TargetSpeedController(config(kp=10.0, ki=10.0))
    assert update(controller, target=2.0, measured=0.0) <= 0.10

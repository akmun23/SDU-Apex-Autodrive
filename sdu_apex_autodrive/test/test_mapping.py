import math

import pytest

from sdu_apex_autodrive.mapping import (
    ForwardSpeedController,
    InterfaceConfig,
    SpeedControllerConfig,
    bound_target_speed,
    steering_angle_to_command,
)


def test_steering_uses_documented_normalized_range():
    config = InterfaceConfig(max_steering_angle_rad=0.5236)
    assert steering_angle_to_command(0.0, config) == 0.0
    assert steering_angle_to_command(0.2618, config) == pytest.approx(0.5)
    assert steering_angle_to_command(0.5236, config) == pytest.approx(1.0)
    assert steering_angle_to_command(-0.5236, config) == pytest.approx(-1.0)


def test_steering_and_target_speed_are_bounded():
    config = InterfaceConfig(max_target_speed_mps=1.0)
    assert steering_angle_to_command(5.0, config) == 1.0
    assert steering_angle_to_command(-5.0, config) == -1.0
    assert bound_target_speed(5.0, config) == 1.0
    assert bound_target_speed(-5.0, config) == 0.0


def test_forward_speed_controller_is_bounded_and_keeps_hold_throttle():
    interface = InterfaceConfig()
    controller = ForwardSpeedController(
        SpeedControllerConfig(
            feedforward_gain=0.04,
            proportional_gain=0.2,
            integral_gain=0.1,
            maximum_forward_throttle=0.1,
            throttle_rise_rate_per_sec=10.0,
            throttle_fall_rate_per_sec=10.0,
        ),
        interface,
    )
    assert controller.update(1.0, 0.0, 0.05) == pytest.approx(0.1)
    hold = controller.update(1.0, 1.1, 0.05)
    assert 0.0 < hold < 0.1
    assert controller.integral < 0.0
    assert controller.update(0.0, 0.5, 0.05) == 0.0


def test_forward_speed_controller_accumulates_only_below_cap():
    interface = InterfaceConfig()
    controller = ForwardSpeedController(
        SpeedControllerConfig(
            feedforward_gain=0.0,
            proportional_gain=0.01,
            integral_gain=0.01,
            integral_limit=0.5,
            maximum_forward_throttle=0.1,
            throttle_rise_rate_per_sec=10.0,
            throttle_fall_rate_per_sec=10.0,
        ),
        interface,
    )
    first = controller.update(1.0, 0.0, 0.1)
    second = controller.update(1.0, 0.0, 0.1)
    assert 0.0 < first < second < 0.1
    assert controller.integral == pytest.approx(0.2)


def test_forward_speed_controller_slew_limits_throttle_changes():
    interface = InterfaceConfig()
    controller = ForwardSpeedController(
        SpeedControllerConfig(
            feedforward_gain=0.04,
            proportional_gain=0.02,
            integral_gain=0.0,
            maximum_forward_throttle=0.1,
            throttle_rise_rate_per_sec=0.1,
            throttle_fall_rate_per_sec=0.2,
        ),
        interface,
    )

    outputs = [controller.update(2.0, 0.0, 0.05) for _ in range(5)]
    assert outputs == pytest.approx([0.005, 0.01, 0.015, 0.02, 0.025])
    reduced = controller.update(0.5, 2.0, 0.05)
    assert reduced == pytest.approx(0.015)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_conversion_rejects_nonfinite_commands(value):
    config = InterfaceConfig()
    with pytest.raises(ValueError):
        steering_angle_to_command(value, config)
    with pytest.raises(ValueError):
        bound_target_speed(value, config)

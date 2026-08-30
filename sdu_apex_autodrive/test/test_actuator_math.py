import math

import pytest

from sdu_apex_autodrive.actuator_math import (
    ActuatorLimits,
    bound_target_speed,
    steering_angle_to_normalized,
)


LIMITS = ActuatorLimits()


def test_zero_steering_maps_to_zero():
    assert steering_angle_to_normalized(0.0, LIMITS) == 0.0


def test_steering_limits_map_to_native_limits():
    assert steering_angle_to_normalized(0.5236, LIMITS) == pytest.approx(1.0)
    assert steering_angle_to_normalized(-0.5236, LIMITS) == pytest.approx(-1.0)


def test_over_limit_steering_clamps():
    assert steering_angle_to_normalized(10.0, LIMITS) == 1.0
    assert steering_angle_to_normalized(-10.0, LIMITS) == -1.0


def test_nonfinite_steering_rejects():
    with pytest.raises(ValueError):
        steering_angle_to_normalized(math.nan, LIMITS)


def test_target_speed_clamps():
    assert bound_target_speed(-1.0, LIMITS) == 0.0
    assert bound_target_speed(2.0, LIMITS) == 2.0
    assert bound_target_speed(10.0, LIMITS) == 4.0

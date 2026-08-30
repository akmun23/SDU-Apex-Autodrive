import math

import pytest

from sdu_apex_autodrive.speed_controller import interpolate_feedforward


SPEED = (0.0, 1.0, 2.0)
THROTTLE = (0.0, 0.04, 0.07)


def test_exact_points():
    assert interpolate_feedforward(1.0, SPEED, THROTTLE) == pytest.approx(0.04)


def test_interpolation():
    assert interpolate_feedforward(1.5, SPEED, THROTTLE) == pytest.approx(0.055)


def test_endpoint_clamping():
    assert interpolate_feedforward(-1.0, SPEED, THROTTLE) == 0.0
    assert interpolate_feedforward(10.0, SPEED, THROTTLE) == 0.07


def test_zero_placeholder():
    assert interpolate_feedforward(2.0, (0.0,), (0.0,)) == 0.0


@pytest.mark.parametrize(
    'speeds,throttles',
    [
        ((0.0, 1.0), (0.0,)),
        ((0.0, 1.0, 0.5), (0.0, 0.1, 0.2)),
        ((0.0, math.nan), (0.0, 0.1)),
        ((0.0, 1.0), (0.0, math.inf)),
    ],
)
def test_invalid_tables_reject(speeds, throttles):
    with pytest.raises(ValueError):
        interpolate_feedforward(0.5, speeds, throttles)

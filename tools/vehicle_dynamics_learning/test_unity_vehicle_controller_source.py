import pytest

from tools.vehicle_dynamics_learning.unity_vehicle_controller_source import (
    center_angle_from_front_ackermann_deg,
    front_ackermann_angles_deg,
    step_autonomous_steering_deg,
)


def test_autonomous_steering_source_ramps_and_clamps_target():
    angle = step_autonomous_steering_deg(0.0, 0.0, -1.0, 0.001)
    assert angle == pytest.approx(0.183345, abs=2e-6)
    angle = step_autonomous_steering_deg(angle, angle, -1.0, 0.001)
    assert angle == pytest.approx(0.36669, abs=4e-6)
    assert step_autonomous_steering_deg(
        29.9, 29.9, -1.0, 0.001) == 30.0


def test_source_comparison_can_snap_when_command_is_reduced():
    # With a positive prior internal angle and zero command, Unity takes the
    # increment branch and immediately clamps to the zero target.
    assert step_autonomous_steering_deg(10.0, 10.0, 0.0, 0.001) == 0.0


@pytest.mark.parametrize("center", (-30.0, -17.25, 0.0, 11.5, 30.0))
def test_ackermann_source_formula_round_trips_center_angle(center):
    left, right = front_ackermann_angles_deg(center)
    assert center_angle_from_front_ackermann_deg(left, right) == pytest.approx(
        center, abs=2e-5)


def test_out_of_range_command_is_clamped_like_serialized_range():
    assert step_autonomous_steering_deg(0.0, 0.0, 2.0, 0.001) == pytest.approx(
        -0.183345, abs=2e-6)

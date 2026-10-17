import math

import pytest

from tools.vehicle_dynamics_learning.physx_rigidbody_source import (
    integrate_vehicle_body_substep,
    integrate_free_solver_pose,
    unconstrained_velocity_step,
)


def test_gravity_is_added_before_linear_damping():
    linear, angular = unconstrained_velocity_step(
        (2.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        (0.0, -9.81, 0.0), 0.001, 0.273, 0.1)
    expected_y = (-9.81 * 0.001) * (1.0 - 0.273 * 0.001)
    assert linear == pytest.approx((2.0 * (1.0 - 0.273 * 0.001),
                                    expected_y, 0.0), abs=1e-7)
    assert angular == pytest.approx((0.0, 0.0, 1.0 * (1.0 - 0.1 * 0.001)),
                                    abs=2e-8)


def test_disabled_gravity_leaves_velocity_unchanged_except_damping():
    linear, _ = unconstrained_velocity_step(
        (0.0, 3.0, 0.0), (0.0, 0.0, 0.0),
        (0.0, -9.81, 0.0), 0.01, 0.5, 0.0, disable_gravity=True)
    assert linear == pytest.approx((0.0, 3.0 * 0.995, 0.0), abs=2e-7)


def test_angular_velocity_is_clamped_by_vector_norm_after_damping():
    _, angular = unconstrained_velocity_step(
        (0.0, 0.0, 0.0), (0.0, 0.0, 10.0),
        (0.0, 0.0, 0.0), 0.001, 0.0, 0.1)
    assert math.sqrt(sum(value * value for value in angular)) == pytest.approx(
        7.0, abs=1e-6)


def test_linear_velocity_clamp_uses_squared_threshold():
    linear, _ = unconstrained_velocity_step(
        (3.0, 4.0, 0.0), (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0), 0.001, 0.0, 0.0,
        max_linear_velocity_sq=4.0)
    assert linear == pytest.approx((1.2, 1.6, 0.0), abs=1e-7)


def test_damping_multiplier_floors_at_zero():
    linear, angular = unconstrained_velocity_step(
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0), 1.0, 2.0, 2.0)
    assert linear == (0.0, 0.0, 0.0)
    assert angular == (0.0, 0.0, 0.0)


def test_vehicle_body_substep_uses_semi_implicit_linear_integration():
    position, rotation, velocity, angular = integrate_vehicle_body_substep(
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        0.5, (1.0, 1.0, 1.0), 0.1)
    assert velocity == pytest.approx((1.1, 0.0, 0.0), abs=1e-7)
    assert position == pytest.approx((0.11, 0.0, 0.0), abs=1e-7)
    assert rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-7)
    assert angular == pytest.approx((0.0, 0.0, 0.0), abs=1e-7)


def test_vehicle_body_substep_rotates_diagonal_inertia_to_world_frame():
    half = math.sqrt(0.5)
    _, _, _, angular = integrate_vehicle_body_substep(
        (0.0, 0.0, 0.0), (0.0, 0.0, half, half),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        1.0, (2.0, 1.0, 0.5), 0.01)
    # A 90-degree body rotation maps body Y's inverse inertia (1) to
    # world X; the result must not use body X's value (2).
    assert angular == pytest.approx((0.01, 0.0, 0.0), abs=1e-7)


def test_vehicle_body_substep_quaternion_uses_world_angular_velocity():
    _, rotation, _, _ = integrate_vehicle_body_substep(
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0), (0.0, 0.0, 2.0),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        1.0, (1.0, 1.0, 1.0), 0.1)
    assert rotation == pytest.approx((0.0, 0.0, 0.0995037, 0.9950372),
                                     abs=2e-6)
    assert math.sqrt(sum(value * value for value in rotation)) == pytest.approx(
        1.0, abs=1e-7)


def test_scene_solver_pose_uses_closed_form_quaternion_integrator():
    position, rotation = integrate_free_solver_pose(
        (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0),
        (2.0, 0.0, 0.0), (0.0, 0.0, 2.0), 0.1)
    assert position == pytest.approx((1.2, 2.0, 3.0), abs=1e-7)
    assert rotation == pytest.approx((0.0, 0.0, math.sin(0.1),
                                      math.cos(0.1)), abs=1e-7)


def test_scene_solver_pose_with_zero_angular_velocity_preserves_orientation():
    original = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    position, rotation = integrate_free_solver_pose(
        (0.0, 0.0, 0.0), original, (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0), 0.001)
    assert position == pytest.approx((0.0, 0.001, 0.0), abs=1e-7)
    assert rotation == pytest.approx(original, abs=1e-7)


@pytest.mark.parametrize("kwargs", [
    {"dt_s": 0.0},
    {"linear_damping_per_s": -0.1},
    {"max_linear_velocity_sq": -1.0},
    {"max_angular_velocity_radps": float("nan")},
])
def test_invalid_rigidbody_parameters_are_rejected(kwargs):
    parameters = dict(
        linear_velocity_mps=(0.0, 0.0, 0.0),
        angular_velocity_radps=(0.0, 0.0, 0.0),
        gravity_mps2=(0.0, -9.81, 0.0),
        dt_s=0.001,
        linear_damping_per_s=0.273,
        angular_damping_per_s=0.1,
    )
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        unconstrained_velocity_step(**parameters)

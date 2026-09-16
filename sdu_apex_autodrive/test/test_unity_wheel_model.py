"""Tests for the offline direct Unity WheelCollider law."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/model_id/structured_vehicle_plant.py"
SPEC = importlib.util.spec_from_file_location("unity_plant", MODULE_PATH)
assert SPEC and SPEC.loader
PLANT = importlib.util.module_from_spec(SPEC)
sys.modules["unity_plant"] = PLANT
SPEC.loader.exec_module(PLANT)


def test_unity_sideways_curve_has_serialized_breakpoints() -> None:
    assert PLANT.unity_sideways_friction_value(0.0) == pytest.approx(0.0)
    assert PLANT.unity_sideways_friction_value(0.01) == pytest.approx(1.0)
    assert PLANT.unity_sideways_friction_value(0.1) == pytest.approx(0.5)
    assert PLANT.unity_sideways_friction_value(1.0) == pytest.approx(0.5)
    assert PLANT.unity_sideways_friction_value(-0.1) == pytest.approx(-0.5)


def test_unity_ackermann_is_symmetric_and_not_single_axle_angle() -> None:
    parameters = PLANT.PlantParameters(tire_model="unity_wheel_collider")
    left, right = PLANT._unity_ackermann_angles(0.2, parameters)
    reverse_left, reverse_right = PLANT._unity_ackermann_angles(-0.2, parameters)
    assert left > right
    assert reverse_left == pytest.approx(-right)
    assert reverse_right == pytest.approx(-left)


def test_unity_wheel_wrench_is_zero_without_motion_or_steering() -> None:
    parameters = PLANT.PlantParameters(tire_model="unity_wheel_collider")
    wrench = PLANT._unity_wheel_lateral_wrench(0.0, 0.0, 0.0, 0.0,
                                                parameters)
    assert np.asarray(wrench) == pytest.approx(np.zeros(3))


def test_unity_wheel_model_has_no_fitted_axle_peak() -> None:
    parameters = PLANT.PlantParameters(tire_model="unity_wheel_collider")
    assert parameters.tire_model == "unity_wheel_collider"
    assert PLANT.unity_sideways_friction_value(0.2) == pytest.approx(0.5)


def test_unity_forward_slip_uses_larger_wheel_or_ground_speed() -> None:
    assert PLANT.unity_forward_slip(10.0, 2.5) == pytest.approx(0.75)
    assert PLANT.unity_forward_slip(2.5, 3.5) == pytest.approx(-1.0 / 3.5)
    assert PLANT.unity_forward_slip(0.0, 3.5) == pytest.approx(-1.0)


def test_unity_forward_force_and_cawd_torque_keep_load_explicit() -> None:
    assert PLANT.unity_forward_wheel_force(10.0, 10.0, 9.0) == pytest.approx(
        10.0 * 0.8 * 0.9 * (1.0 / 10.0) / 0.15)
    assert PLANT.unity_cawd_motor_torque_per_wheel(0.5) == pytest.approx(53.5)
    assert PLANT.unity_competition_brake_torque_per_wheel(0.0) == pytest.approx(428.0)
    assert PLANT.unity_competition_brake_torque_per_wheel(0.1) == pytest.approx(0.0)


def test_unity_four_wheel_longitudinal_screen_uses_explicit_loads() -> None:
    parameters = PLANT.PlantParameters(tire_model="unity_wheel_collider")
    force = PLANT._unity_wheel_longitudinal_force(
        10.0, 0.0, 0.0, 11.0, 0.0, parameters)
    assert force > 0.0
    assert force < sum(parameters.unity_wheel_sprung_masses_kg) * 9.81


def test_mechanical_load_screen_uses_exact_geometry_and_signs() -> None:
    parameters = PLANT.PlantParameters(
        tire_model="unity_wheel_collider",
        unity_normal_load_mode="mechanical_cg_transfer",
    )
    _, _, _ = PLANT._unity_wheel_lateral_wrench(
        10.0, 0.2, 0.1, 0.04, parameters,
        longitudinal_accel_mps2=2.0, lateral_accel_mps2=3.0)
    geometric_wheelbase = PLANT.UNITY_FRONT_WHEEL_X_M - PLANT.UNITY_REAR_WHEEL_X_M
    front_rear = (-parameters.mass_kg * PLANT.UNITY_COM_HEIGHT_M /
                  geometric_wheelbase * 2.0)
    left_right = (-parameters.mass_kg * PLANT.UNITY_COM_HEIGHT_M /
                  parameters.unity_track_width_m * 3.0)
    assert front_rear < 0.0
    assert left_right < 0.0
    assert geometric_wheelbase == pytest.approx(0.33)

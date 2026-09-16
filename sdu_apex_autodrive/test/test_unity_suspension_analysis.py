import numpy as np
import pandas as pd
import pytest

from tools.model_id.analyze_unity_suspension_response import (
    _derivative,
    _reset_boundaries,
    _reset_exclusion_mask,
)
from tools.model_id.identify_unity_axle_force_response import (
    _derivative as _axle_derivative,
    _reset_boundaries as _axle_reset_boundaries,
    _reset_exclusion_mask as _axle_reset_exclusion_mask,
)
from tools.model_id.analyze_unity_wheel_contact_trace import (
    _wheel_forward_slip_candidates,
)


def _row(x: float, speed: float) -> dict[str, str]:
    return {
        "root_position_x_m": str(x),
        "root_position_y_m": "0.1",
        "root_position_z_m": "0.0",
        "body_velocity_x_mps": "0.0",
        "body_velocity_y_mps": "0.0",
        "body_velocity_z_mps": str(speed),
    }


def test_reset_segmentation_detects_teleport_to_stationary_pose():
    rows = [_row(0.0, 3.0), _row(0.003, 3.0), _row(-1.0, 0.0),
            _row(-0.997, 0.1)]
    boundaries = _reset_boundaries(rows)
    assert boundaries.tolist() == [False, False, True, False]
    assert _reset_exclusion_mask(boundaries, radius=1).tolist() == [
        False, True, True, True]


def test_segmented_derivative_never_uses_reset_jump():
    values = np.asarray([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    times = np.arange(len(values), dtype=float)
    boundaries = np.asarray([False, False, False, True, False, False])
    derivative = _derivative(values, times, boundaries)
    assert np.allclose(derivative[[0, 1, 2, 4, 5]], 1.0)
    assert np.isnan(derivative[3])


def test_axle_force_inversion_segments_reset_derivatives():
    frame = pd.DataFrame({
        "world_com_x_m": [0.0, 0.003, -1.0, -0.997],
        "world_com_y_m": [0.1, 0.1, 0.1, 0.1],
        "world_com_z_m": [0.0, 0.003, 0.0, 0.003],
        "world_velocity_x_mps": [0.0, 3.0, 0.0, 0.1],
        "world_velocity_y_mps": [0.0, 0.0, 0.0, 0.0],
        "world_velocity_z_mps": [3.0, 3.0, 0.0, 0.1],
    })
    boundaries = _axle_reset_boundaries(frame)
    assert boundaries.tolist() == [False, False, True, False]
    times = np.arange(len(frame), dtype=float)
    values = np.asarray([0.0, 1.0, 100.0, 101.0])
    derivative = _axle_derivative(values, times, boundaries)
    assert np.isnan(derivative[2])
    assert _axle_reset_exclusion_mask(boundaries, radius=1).tolist() == [
        False, True, True, True]


def test_forward_slip_candidate_uses_recorded_direction_and_wheel_radius():
    radius = 0.059
    surface_speed = 3.3
    row = {
        "world_velocity_x_mps": "0",
        "world_velocity_y_mps": "0",
        "world_velocity_z_mps": "3",
        "world_angular_velocity_x_radps": "0",
        "world_angular_velocity_y_radps": "0",
        "world_angular_velocity_z_radps": "0",
        "world_com_x_m": "0",
        "world_com_y_m": "0",
        "world_com_z_m": "0",
        "wheel0_world_pose_x_m": "0.17",
        "wheel0_world_pose_y_m": "0",
        "wheel0_world_pose_z_m": "0",
        "wheel0_contact_point_x_m": "0.17",
        "wheel0_contact_point_y_m": "-0.059",
        "wheel0_contact_point_z_m": "0",
        "wheel0_forward_dir_x": "0",
        "wheel0_forward_dir_y": "0",
        "wheel0_forward_dir_z": "1",
        "wheel0_rpm": str(surface_speed / radius * 60.0 / (2.0 * np.pi)),
        "wheel0_forward_slip": "0.1",
    }
    recorded, candidates = _wheel_forward_slip_candidates(row, 0, radius)
    assert recorded == pytest.approx(0.1)
    assert candidates[
        "wheel_center_wheel_minus_ground_abs_ground_floor_0p25"] == pytest.approx(
            0.1)
    assert candidates[
        "wheel_center_ground_minus_wheel_abs_ground_floor_0p25"] == pytest.approx(
            -0.1)

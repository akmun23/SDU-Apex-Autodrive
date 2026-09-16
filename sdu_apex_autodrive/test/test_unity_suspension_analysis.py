import numpy as np
import pandas as pd

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

"""Numerical contract checks for segmented plant-training horizons."""

import pytest

from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    _horizon_choices,
    _validation_score,
)


def test_training_sections_are_shorter_than_failed_five_second_bptt() -> None:
    assert _horizon_choices("L0") == (1,)
    assert _horizon_choices("L1") == (4, 10, 20)
    assert _horizon_choices("L2") == (20, 40, 80)
    assert _horizon_choices("L3") == (120, 160, 200)
    assert _horizon_choices("L4") == (240, 320, 400)
    assert max(_horizon_choices("L2")) * 0.025 == 2.0
    assert max(_horizon_choices("L3")) * 0.025 == 5.0
    assert max(_horizon_choices("L4")) * 0.025 == 10.0


def test_validation_score_is_unitless_and_equal_weighted() -> None:
    names = (
        "position_radial_trajectory_rmse_m",
        "heading_trajectory_rmse_rad",
        "u_rmse_mps",
        "v_rmse_mps",
        "yaw_rate_rmse_rps",
    )
    parent = {name: 2.0 for name in names}
    candidate = {name: 1.0 for name in names}
    score, ratios = _validation_score(candidate, parent)
    assert score == 0.5
    assert all(value == 0.5 for value in ratios.values())


def test_validation_score_rejects_missing_physical_channels() -> None:
    with pytest.raises(KeyError):
        _validation_score({"u_rmse_mps": 1.0}, {"u_rmse_mps": 1.0})

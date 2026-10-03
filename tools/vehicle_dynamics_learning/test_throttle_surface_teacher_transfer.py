#!/usr/bin/env python3
"""Focused checks for source-aligned wheel-rate transfer labels."""

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import WHEEL_RADIUS_M
from tools.vehicle_dynamics_learning.score_throttle_surface_teacher_transfer import (
    HORIZONS,
    _metric_row,
    _source_aligned_encoder_rates,
    _steering_angle_breakdown,
)


def test_raw_encoder_rates_are_causal_and_do_not_cross_sequences():
    packet = np.asarray((0, 1, 2, 4, 0, 1, 2, 3), dtype=np.int64)
    stamps = np.asarray((0, 25, 65, 90, 100, 125, 150, 175), dtype=np.int64) * 1_000_000
    position = np.zeros((8, 2), dtype=np.float64)
    position[1] = 0.025 / WHEEL_RADIUS_M
    position[2] = position[1] + 0.040 / WHEEL_RADIUS_M
    position[3] = position[2] + 0.025 / WHEEL_RADIUS_M
    position[5] = 0.025 / WHEEL_RADIUS_M
    position[6] = position[5] + 0.025 / WHEEL_RADIUS_M
    position[7] = position[6] + 0.025 / WHEEL_RADIUS_M
    matched = np.ones((8, 2), dtype=bool)
    matched[6, 1] = False

    rates, valid = _source_aligned_encoder_rates(
        position, matched, packet, stamps, np.asarray(((0, 4), (4, 8))))

    np.testing.assert_array_equal(
        valid, np.asarray((False, True, False, False, False, True, False, False)))
    np.testing.assert_allclose(rates[1], (1.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(rates[5], (1.0, 1.0), atol=1e-7)
    assert np.isnan(rates[0]).all()
    assert np.isnan(rates[4]).all()


def test_rollout_horizons_and_signed_steering_breakdown_are_preserved():
    assert HORIZONS["0.025s"] == 1
    assert HORIZONS["5s"] == 200
    rows = [
        {"angle_cluster": -0.2, "high_throttle": False,
         "initial_speed_mps": 3.0, "mse": {"wheel": 1.0}},
        {"angle_cluster": -0.2, "high_throttle": True,
         "initial_speed_mps": 5.0, "mse": {"wheel": 9.0}},
        {"angle_cluster": 0.2, "high_throttle": True,
         "initial_speed_mps": 4.0, "mse": {"wheel": 4.0}},
    ]

    result = _steering_angle_breakdown(rows, ["wheel"])

    assert result["-0.2000"]["condition_count"] == 2
    assert result["-0.2000"]["high_throttle_condition_count"] == 1
    assert result["-0.2000"]["start_speed_mps_mean"] == 4.0
    assert result["-0.2000"]["rmse"]["wheel"] == np.sqrt(5.0)
    assert result["-0.2000"]["high_throttle_rmse"]["wheel"] == 3.0
    assert result["0.2000"]["rmse"]["wheel"] == 2.0


def test_first_horizon_remains_scoreable_when_raw_wheel_packet_is_invalid():
    state = np.zeros((1, 7), dtype=np.float64)
    pose = np.zeros((2, 3), dtype=np.float64)

    metrics = _metric_row(state, state, pose, pose, 1,
                          wheel_valid=np.asarray((False,)))

    assert metrics["u_com_mps"] == 0.0
    assert metrics["rear_left_wheel_mps"] is None
    assert metrics["rear_right_wheel_mps"] is None

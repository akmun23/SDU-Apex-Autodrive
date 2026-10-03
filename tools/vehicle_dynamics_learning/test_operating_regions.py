from __future__ import annotations

import numpy as np
import pytest

from tools.vehicle_dynamics_learning.operating_regions import (
    region_masks,
    steering_sign,
)


def test_region_boundaries_and_signed_steering_are_canonical() -> None:
    frames = np.zeros((6, 9), dtype=np.float64)
    frames[:, 0] = (2.999, 3.0, 7.0, 8.999, 9.0, 10.5)
    frames[:, 3] = (0.0, 0.10, -0.30, 0.524, -0.20, 0.05)
    frames[:, 8] = (0.0, -0.06, 0.10, 0.60, 0.20, 0.0)
    resets = np.zeros(6, dtype=np.int64)
    packets = np.arange(100, 106, dtype=np.int64)
    masks = region_masks(frames, resets, packets, np.ones(6, dtype=bool))

    assert masks["S0"].tolist() == [True, False, False, False, False, False]
    assert masks["S1"][1]
    assert masks["S3"][3]
    assert masks["S4"][4]
    assert masks["S5"][5]
    assert masks["D1"][1]
    assert masks["D3"][2]
    assert masks["D4"][3]
    assert masks["negative_command_braking"][1]
    assert masks["near_zero_throttle_command"][0]
    assert masks["low_positive_throttle_command"][2]
    assert masks["high_throttle_command"][3]
    assert steering_sign(frames[:, 3]).tolist() == [0, 1, -1, 1, -1, 1]


def test_rate_and_transition_masks_do_not_cross_packet_gaps_or_resets() -> None:
    frames = np.zeros((4, 9), dtype=np.float64)
    frames[:, 0] = 5.0
    frames[:, 3] = (0.0, 0.12, 0.20, 0.40)
    frames[:, 8] = (0.0, 0.01, 0.02, 0.40)
    reset = np.asarray([0, 0, 0, 1])
    packet = np.asarray([10, 11, 13, 14])
    masks = region_masks(frames, reset, packet, np.ones(4, dtype=bool))

    assert masks["steering_turn_in"][1]
    assert not masks["steering_turn_in"][2]
    assert not masks["steering_turn_in"][3]
    assert not masks["simultaneous_steering_throttle_transition"][2]
    assert not masks["simultaneous_steering_throttle_transition"][3]


def test_nonfinite_rows_and_misaligned_inputs_are_rejected() -> None:
    frames = np.zeros((2, 9), dtype=np.float64)
    frames[1, 0] = np.nan
    masks = region_masks(frames, np.zeros(2), np.arange(2), np.ones(2, bool))
    assert all(not mask[1] for mask in masks.values())
    with pytest.raises(ValueError, match="aligned nine-channel"):
        region_masks(frames, np.zeros(1), np.arange(2), np.ones(2, bool))
    with pytest.raises(ValueError, match="steering feedback must be finite"):
        steering_sign(np.asarray([0.0, np.nan]))

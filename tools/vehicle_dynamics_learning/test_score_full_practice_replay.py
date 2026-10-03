#!/usr/bin/env python3
"""Focused signal-alignment check for full-capture replay metrics."""

import numpy as np

from tools.vehicle_dynamics_learning.score_full_practice_replay import (
    _command_indices,
    _match_isolated_lap_crossing,
    _metrics,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _window_batch,
)
from tools.vehicle_dynamics_learning.diagnose_encoder_cadence_alignment import (
    _condition_masks,
    encoder_rate_pair,
)


def test_raw_wheel_metrics_exclude_invalid_encoder_intervals_per_side():
    predicted = np.zeros((3, 7), dtype=np.float64)
    truth = np.zeros_like(predicted)
    truth[:, 5] = (2.0, 100.0, 2.0)
    truth[:, 6] = (4.0, 100.0, 4.0)
    pose = np.zeros((3, 3), dtype=np.float64)

    result = _metrics(predicted, pose, truth, pose,
                      wheel_valid=np.asarray((True, False, True)))

    assert result["wheel_valid_sample_count"] == 2
    assert result["rear_left_wheel_rmse_mps"] == 2.0
    assert result["rear_right_wheel_rmse_mps"] == 4.0
    assert result["rear_wheel_pair_rmse_mps"] == np.sqrt(10.0)
    assert result["speed_rmse_mps"] == 0.0


def test_pose_error_is_resolved_in_truth_track_frame():
    state = np.zeros((2, 7), dtype=np.float64)
    truth_pose = np.asarray(((0.0, 0.0, np.pi / 2),) * 2)
    predicted_pose = truth_pose.copy()
    predicted_pose[:, 1] = 2.0

    result = _metrics(state, predicted_pose, state, truth_pose)

    assert result["along_track_rmse_m"] == 2.0
    assert result["along_track_bias_m"] == 2.0
    assert result["cross_track_rmse_m"] < 1e-12


def test_isolated_lap_crossing_ignores_reinitialized_gate_crossing():
    crossing = _match_isolated_lap_crossing([0.002, 5.9, 11.9], 6.0)

    assert crossing == 5.9
    assert _match_isolated_lap_crossing([0.002], 6.0) is None


def test_one_frame_command_alignment_uses_command_preceding_target_state():
    target_rows = np.asarray((101, 102, 103))

    np.testing.assert_array_equal(_command_indices(target_rows, -1),
                                  (100, 101, 102))
    np.testing.assert_array_equal(_command_indices(target_rows, 0),
                                  target_rows)


def test_validation_window_commands_can_follow_fitted_one_frame_delay():
    frames = np.zeros((100, 9), dtype=np.float32)
    frames[:, 7] = np.arange(100)
    frames[:, 8] = np.arange(100) + 100
    data = {
        "frames": frames,
        "simulator_pose_xyyaw": np.zeros((100, 3), dtype=np.float32),
    }
    state = np.zeros((100, 7), dtype=np.float32)

    batch = _window_batch(
        data, state, [{"start": 80}], history_state_size=7,
        horizon_steps=2, command_offset_frames=-1)

    np.testing.assert_array_equal(batch[3][0, :, 0], (80, 81))
    np.testing.assert_array_equal(batch[3][0, :, 1], (180, 181))
    np.testing.assert_array_equal(batch[4][0, :, 0], (0, 0))


def test_encoder_cadence_diagnostic_compares_fixed_25ms_with_stamp_interval():
    previous = (1_000_000_000, 0.0)
    current = (1_020_000_000, 0.1)

    stamped, fixed, dt_s = encoder_rate_pair(previous, current)

    assert np.isclose(dt_s, 0.020)
    assert np.isclose(stamped, 0.059 * 0.1 / 0.020)
    assert np.isclose(fixed, 0.059 * 0.1 / 0.025)


def test_encoder_cadence_condition_masks_keep_one_value_per_frame():
    frames = np.zeros((4, 9), dtype=np.float64)
    rigid = np.zeros((4, 13), dtype=np.float64)
    rigid[:, 7] = (1.0, 2.0, 3.0, 4.0)
    dt_s = np.full(4, 0.025)
    same_epoch = np.asarray((False, True, True, True))

    masks = _condition_masks(frames, rigid, dt_s, same_epoch)

    assert all(mask.shape == (4,) for mask in masks.values())

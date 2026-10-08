"""Focused causal-window and rollout-shape tests for the yaw GRU teacher."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tools.racing.specialists.train_yaw_gru_trajectory_teacher import (
    FUTURE_STEPS,
    HISTORY_STEPS,
    RunWindows,
    _exclude_validation_runs,
    gather_windows,
    make_model,
    window_starts,
)


class YawTrajectoryWindowTest(unittest.TestCase):
    def test_heldout_validation_runs_can_be_excluded_from_selection(self):
        series = [
            SimpleNamespace(run_id="train-r01", split="train"),
            SimpleNamespace(run_id="old-validation", split="validation"),
            SimpleNamespace(run_id="heldout-r03", split="validation"),
        ]
        remaining = _exclude_validation_runs(series, {"heldout-r03"})
        self.assertEqual([row.run_id for row in remaining],
                         ["train-r01", "old-validation"])

    def test_only_discovered_validation_runs_can_be_excluded(self):
        series = [SimpleNamespace(run_id="train-r01", split="train")]
        with self.assertRaisesRegex(ValueError, "not discovered"):
            _exclude_validation_runs(series, {"unknown-r03"})
        with self.assertRaisesRegex(ValueError, "only validation runs"):
            _exclude_validation_runs(series, {"train-r01"})

    def test_window_inputs_and_targets_are_causal_and_reset_bounded(self):
        observations = np.arange(2 * 140 * 11, dtype=np.float32).reshape(-1, 11)
        commands = np.arange(2 * 140 * 2, dtype=np.float32).reshape(-1, 2)
        yaw = np.arange(2 * 140, dtype=np.float32)
        valid = np.ones(len(yaw), dtype=bool)
        starts = window_starts(
            np.asarray(((0, 140), (140, 280))), valid, valid, valid)
        run = RunWindows("toy", "train", observations, commands, yaw, starts)
        current = int(starts[0])
        past, future, target = gather_windows(run, np.asarray([current]))

        self.assertEqual(past.shape, (1, HISTORY_STEPS, 11))
        self.assertEqual(future.shape, (1, FUTURE_STEPS, 2))
        self.assertEqual(target.shape, (1, FUTURE_STEPS))
        np.testing.assert_array_equal(
            past[0], observations[current - HISTORY_STEPS + 1:current + 1])
        np.testing.assert_array_equal(
            future[0], commands[current:current + FUTURE_STEPS])
        np.testing.assert_array_equal(
            target[0], yaw[current + 1:current + FUTURE_STEPS + 1])
        self.assertTrue(np.any(starts < 140))
        self.assertTrue(np.any(starts >= 140))
        self.assertFalse(np.any((starts < 140)
                                & (starts + FUTURE_STEPS >= 140)))

    def test_invalid_history_or_future_command_rejects_only_crossing_windows(self):
        length = 180
        valid_obs = np.ones(length, dtype=bool)
        valid_commands = np.ones(length, dtype=bool)
        valid_yaw = np.ones(length, dtype=bool)
        valid_commands[110] = False
        starts = window_starts(
            np.asarray(((0, length),)), valid_obs, valid_commands, valid_yaw)
        self.assertNotIn(110, starts)
        self.assertIn(111, starts)
        valid_obs[90] = False
        starts = window_starts(
            np.asarray(((0, length),)), valid_obs, valid_commands, valid_yaw)
        self.assertFalse(np.any((starts >= 90)
                                & (starts <= 90 + HISTORY_STEPS - 1)))


class YawTrajectoryModelTest(unittest.TestCase):
    def test_encoder_decoder_returns_all_candidate_horizons(self):
        import torch

        model = make_model()
        past = torch.zeros(3, HISTORY_STEPS, 11)
        future = torch.zeros(3, FUTURE_STEPS, 2)
        prediction = model(past, future)
        self.assertEqual(tuple(prediction.shape), (3, FUTURE_STEPS))
        self.assertTrue(torch.isfinite(prediction).all().item())


if __name__ == "__main__":
    unittest.main()

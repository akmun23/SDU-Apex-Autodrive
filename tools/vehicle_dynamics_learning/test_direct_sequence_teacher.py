"""Data-boundary invariants for the direct trajectory ceiling model."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    _eligible_sequences,
    _sample_windows,
    _training_batch,
)


def _small_dataset():
    frames = np.arange(30 * 9, dtype=np.float32).reshape(30, 9)
    return {
        "frames": frames,
        "bounds": np.asarray([[0, 15], [15, 30]], dtype=np.int64),
        "seq_run": np.asarray([0, 1], dtype=np.int32),
        "run_ids": np.asarray(["train_run", "validation_run"]),
        "splits": np.asarray(["train", "validation"]),
    }


class DirectSequenceWindowTests(unittest.TestCase):
    def test_windows_are_whole_run_split_and_drop_only_terminal_derivative(self):
        data = _small_dataset()
        targets = np.zeros((30, 10), dtype=np.float32)
        targets[14, 9] = np.nan
        targets[29, 9] = np.nan

        groups = _eligible_sequences(data, targets, "train", 4, 6)

        self.assertEqual(groups, {0: [(0, 15)]})

    def test_sampled_windows_never_cross_a_sequence_end(self):
        data = _small_dataset()
        targets = np.zeros((30, 10), dtype=np.float32)
        groups = _eligible_sequences(data, targets, "train", 4, 6)
        windows = _sample_windows(
            data, groups, 8, 4, 6, np.random.default_rng(7))

        self.assertEqual(len(windows), 8)
        self.assertTrue(all(start + 4 + 6 <= end - 1
                            for start, end in windows))

    def test_training_batch_uses_only_future_commands_and_targets(self):
        data = _small_dataset()
        targets = np.zeros((30, 10), dtype=np.float32)
        groups = _eligible_sequences(data, targets, "train", 4, 6)
        context, commands, labels = _training_batch(
            data, targets, groups, 3, 4, 6, np.random.default_rng(11))

        self.assertEqual(context.shape, (3, 4, 9))
        self.assertEqual(commands.shape, (3, 6, 2))
        self.assertEqual(labels.shape, (3, 6, 10))
        np.testing.assert_array_equal(commands[0, :, 0] % 9,
                                      np.asarray([7, 7, 7, 7, 7, 7]))


if __name__ == "__main__":
    unittest.main()

"""Causality and indexing tests for matched-state history descriptors."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.matched_state_analysis import (
    _balanced_query_indices,
    _history_features,
)


class HistoryFeatureTests(unittest.TestCase):
    def test_100ms_history_uses_four_past_states_and_current_state(self):
        values = np.repeat(np.arange(8, dtype=np.float64)[:, None], 9, axis=1)
        feature = _history_features(values, np.asarray([4]), 4)[0]

        np.testing.assert_array_equal(feature[:9], np.full(9, 4.0))
        np.testing.assert_array_equal(feature[9:18], np.full(9, 0.0))
        for band, value in enumerate((0.0, 1.0, 2.0, 3.0)):
            np.testing.assert_array_equal(
                feature[18 + band * 9:27 + band * 9], np.full(9, value))

    def test_history_descriptor_does_not_read_future_samples(self):
        original = np.repeat(np.arange(10, dtype=np.float64)[:, None], 9, axis=1)
        changed_future = original.copy()
        changed_future[6:] += 1000.0
        original_features = _history_features(original, np.asarray([5]), 4)
        changed_features = _history_features(changed_future, np.asarray([5]), 4)
        np.testing.assert_array_equal(original_features, changed_features)

    def test_blind_whole_runs_are_not_queries_or_neighbours(self):
        row_count = 220
        frames = np.zeros((row_count, 9), dtype=np.float64)
        frames[:, 0] = 3.0
        rigid = np.zeros((row_count, 13), dtype=np.float64)
        acceleration = np.zeros((row_count, 3), dtype=np.float64)
        data = {
            "sequence_bounds": np.asarray([[0, 110], [110, 220]]),
            "sequence_run_index": np.asarray([0, 1]),
            "run_splits": np.asarray(["train", "test"]),
            "frames": frames,
            "simulator_rigid_state": rigid,
            "simulator_linear_acceleration": acceleration,
            "frame_run_index": np.repeat([0, 1], 110),
        }

        indices, run_ids, _ = _balanced_query_indices(
            data, np.zeros((row_count, 9)), seed=13)

        self.assertGreater(len(indices), 0)
        self.assertTrue(np.all(indices < 110))
        np.testing.assert_array_equal(run_ids, np.zeros(len(run_ids), dtype=np.int32))


if __name__ == "__main__":
    unittest.main()

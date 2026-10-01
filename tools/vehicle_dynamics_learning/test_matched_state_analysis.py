"""Causality and indexing tests for matched-state history descriptors."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.matched_state_analysis import (
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


if __name__ == "__main__":
    unittest.main()

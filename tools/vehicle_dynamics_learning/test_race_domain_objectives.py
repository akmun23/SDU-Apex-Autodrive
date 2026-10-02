import unittest

import numpy as np

from tools.vehicle_dynamics_learning.race_domain_objectives import (
    RACE_SPEED_BIN_EDGES_MPS,
    race_speed_mismatch_weights,
    race_speed_bin_weights,
    race_speed_domain_labels,
)


class RaceDomainObjectiveTests(unittest.TestCase):
    def test_populated_speed_bins_have_equal_total_weight(self):
        speed = np.asarray([1.0] * 8 + [6.0] * 4 + [8.0] * 2 + [10.0])
        weights = race_speed_bin_weights(speed)
        edges = np.asarray(RACE_SPEED_BIN_EDGES_MPS)
        bins = np.searchsorted(edges, speed, side="right") - 1
        totals = [weights[bins == index].sum()
                  for index in np.unique(bins)]
        np.testing.assert_allclose(totals, np.full(4, len(speed) / 4.0))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_speed_weighting_rejects_unsupported_labels(self):
        with self.assertRaises(ValueError):
            race_speed_bin_weights(np.asarray([8.0, 12.01]))

    def test_validation_domains_split_at_nine_meters_per_second(self):
        speed = np.asarray([0.0, 8.999, 9.0, 11.99, 12.0])
        labels = race_speed_domain_labels(speed)
        self.assertEqual(labels.tolist(), [
            "core_0_to_9", "core_0_to_9", "fast_9_to_12",
            "fast_9_to_12", "fast_9_to_12",
        ])

    def test_speed_and_wheel_body_mismatch_are_both_balanced(self):
        speed = np.asarray([1.0] * 8 + [8.0] * 2)
        mismatch = np.asarray([0.05] * 8 + [1.0] * 2)
        weights = race_speed_mismatch_weights(speed, mismatch, (0.2, 0.6))
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreater(float(weights[-1]), float(weights[0]))

    def test_mismatch_weights_require_training_thresholds_and_aligned_data(self):
        with self.assertRaises(ValueError):
            race_speed_mismatch_weights(np.asarray([2.0]), np.asarray([0.1]),
                                        (0.2, 0.1))
        with self.assertRaises(ValueError):
            race_speed_mismatch_weights(np.asarray([2.0]), np.asarray([]),
                                        (0.1, 0.2))


if __name__ == "__main__":
    unittest.main()

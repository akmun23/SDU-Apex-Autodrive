from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.analyze_signed_turn_yaw_error import (
    _summarize_group,
)


class SignedTurnYawSummaryTests(unittest.TestCase):
    def test_metrics_are_macro_averaged_by_independent_run(self):
        accel = np.asarray([[1.0], [1.0], [3.0]])
        yaw = np.asarray([[0.1], [0.1], [0.3]])
        mask = np.ones((3, 1), dtype=bool)
        runs = np.asarray([0, 0, 1])
        result = _summarize_group(
            accel, yaw, mask,
            np.asarray(["run_a", "run_b"]), runs)
        self.assertEqual(result["independent_runs"], 2)
        self.assertEqual(result["transitions"], 3)
        self.assertAlmostEqual(
            result["macro_run"]["yaw_acceleration_bias_rps2"], 2.0)
        self.assertAlmostEqual(
            result["macro_run"]["yaw_rate_bias_rps"], 0.2)

    def test_empty_group_is_explicit(self):
        result = _summarize_group(
            np.zeros((2, 1)), np.zeros((2, 1)),
            np.zeros((2, 1), dtype=bool), np.asarray(["a"]),
            np.asarray([0, 0]))
        self.assertEqual(result["independent_runs"], 0)
        self.assertIsNone(result["macro_run"])


if __name__ == "__main__":
    unittest.main()

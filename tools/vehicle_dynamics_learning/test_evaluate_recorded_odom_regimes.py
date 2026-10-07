#!/usr/bin/env python3
"""Mathematical tests for offline odometry regime scoring."""

from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.evaluate_recorded_odom_regimes import (
    _error_summary,
    _source_interval_summary,
    _score_regimes,
)


class RecordedOdomRegimeTests(unittest.TestCase):
    def test_source_stamp_rate_uses_the_simulator_packet_clock(self) -> None:
        result = _source_interval_summary([
            0, 25_000_000, 50_000_000, 150_000_000, 175_000_000,
        ])

        self.assertEqual(result["within_run_intervals"], 3)
        self.assertEqual(result["long_intervals_gt_60ms"], 1)
        self.assertAlmostEqual(result["median_within_run_ms"], 25.0)
        self.assertAlmostEqual(result["mean_within_run_rate_hz"], 40.0)
        self.assertEqual(result["duplicate_source_stamps"], 0)

    def test_error_summary_uses_estimate_minus_truth_rmse_and_bias(self) -> None:
        truth = np.asarray([[2.0, 0.1, 0.5], [4.0, -0.1, -0.5]])
        estimate = truth + np.asarray([[0.2, 0.1, 0.2], [-0.2, -0.1, -0.2]])

        result = _error_summary(truth, estimate)

        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["u_rmse_mps"], 0.2)
        self.assertAlmostEqual(result["v_rmse_mps"], 0.1)
        self.assertAlmostEqual(result["yaw_rate_rmse_rps"], 0.2)
        self.assertAlmostEqual(result["u_bias_mps"], 0.0)

    def test_regime_rows_use_absolute_speed_and_steering(self) -> None:
        truth = np.asarray([[4.0, 0.1, 0.0], [-4.5, -0.1, 0.0],
                            [6.0, 0.2, 0.1]])
        estimate = truth + np.asarray([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0],
                                       [0.3, 0.2, 0.1]])

        rows = _score_regimes(
            truth, estimate, np.asarray([4.0, 4.5, 6.0]),
            np.asarray([-0.05, 0.09, 0.15]))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["speed_bin_mps"], [3.0, 5.0])
        self.assertEqual(rows[0]["abs_steering_bin_rad"], [0.0, 0.1])
        self.assertEqual(rows[0]["samples"], 2)
        self.assertAlmostEqual(rows[0]["u_rmse_mps"], 0.1)
        self.assertEqual(rows[1]["speed_bin_mps"], [5.0, 7.0])
        self.assertAlmostEqual(rows[1]["u_rmse_mps"], 0.3)

    def test_feedback_radians_and_fine_high_speed_bins_are_preserved(self) -> None:
        steering_feedback_rad = np.asarray([0.4199])

        rows = _score_regimes(
            np.asarray([[8.0, 0.1, 0.0]]),
            np.asarray([[8.1, 0.1, 0.0]]),
            np.asarray([10.5]), steering_feedback_rad)

        self.assertEqual(rows[0]["speed_bin_mps"], [10.0, 11.0])
        self.assertEqual(rows[0]["abs_steering_bin_rad"], [0.4, 0.5241])


if __name__ == "__main__":
    unittest.main()

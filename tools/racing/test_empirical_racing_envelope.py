#!/usr/bin/env python3
"""Focused mathematical checks for the offline measured racing envelope."""

from __future__ import annotations

import unittest

import numpy as np

from tools.racing.build_empirical_racing_envelope import (
    fit_superellipse,
    speed_bin,
    consecutive_mask,
)


class EmpiricalRacingEnvelopeTests(unittest.TestCase):
    def test_speed_bin_edges_are_left_closed_and_last_bin_includes_12(self) -> None:
        values = np.asarray([0.0, 1.999, 2.0, 6.0, 11.0, 12.0])
        np.testing.assert_array_equal(speed_bin(values), [0, 0, 1, 3, 8, 8])

    def test_packet_gap_invalidates_full_central_difference_stencil(self) -> None:
        packet = np.arange(10, dtype=np.int64)
        packet[5:] += 1
        valid = consecutive_mask(np.zeros(10, dtype=np.int8), packet)
        self.assertTrue(valid[2])
        self.assertFalse(valid[3:7].any())
        self.assertTrue(valid[7])
        self.assertFalse(valid[:2].any())
        self.assertFalse(valid[-2:].any())

    def test_run_balanced_superellipse_recovers_circular_frontier(self) -> None:
        x_values = np.repeat(np.asarray([0.1, 0.3, 0.5, 0.7, 0.9]), 20)
        y_values = np.sqrt(1.0 - x_values**2)
        data = {"ax": [], "ay": [], "run": [], "seq": []}
        for run in range(3):
            for condition in range(2):
                direction = 1.0 if condition == 0 else -1.0
                data["ax"].append(x_values.copy())
                data["ay"].append(direction * y_values)
                data["run"].append(np.full(x_values.size, run, dtype=np.int32))
                data["seq"].append(np.full(x_values.size, run * 2 + condition, dtype=np.int32))
        arrays = {key: np.concatenate(values) for key, values in data.items()}
        axes = {
            "ax_accel": 1.0,
            "ax_brake": 1.0,
            "ay_accel_left": 1.0,
            "ay_accel_right": 1.0,
            "ay_brake_left": 1.0,
            "ay_brake_right": 1.0,
        }
        exponent, report = fit_superellipse(arrays, axes, minimum_samples=20)
        self.assertEqual(report["status"], "fit")
        self.assertEqual(report["independent_runs"], 3)
        self.assertAlmostEqual(exponent, 2.0, delta=0.06)


if __name__ == "__main__":
    unittest.main()

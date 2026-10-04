from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.diagnose_roll_state_oracle import (
    _metric_errors,
    _paired_improvement,
    _window_rows,
)


class RollOracleDiagnosticTests(unittest.TestCase):
    def test_position_and_heading_errors_are_separate_from_body_errors(self):
        predicted = np.zeros((1, 2, 9))
        truth = np.zeros_like(predicted)
        predicted[0, 0, 0] = 1.0
        poses = np.asarray([[[3.0, 4.0, np.pi - 0.01], [0.0, 0.0, 0.0]]])
        truth_pose = np.asarray([[[0.0, 0.0, -np.pi + 0.01], [0.0, 0.0, 0.0]]])
        metrics = _metric_errors(predicted, poses, truth, truth_pose)
        self.assertAlmostEqual(metrics["position_2d_m"][0, 0], 5.0)
        self.assertAlmostEqual(metrics["heading_rad"][0, 0], -0.02)
        self.assertAlmostEqual(metrics["forward_speed_u_mps"][0, 0], 1.0)

    def test_bootstrap_compares_paired_independent_run_improvements(self):
        baseline = {"macro_run_rmse": {"u": 2.0}, "per_run_rmse": {
            "r1": {"u": 2.0}, "r2": {"u": 4.0}}}
        oracle = {"macro_run_rmse": {"u": 1.5}, "per_run_rmse": {
            "r1": {"u": 1.0}, "r2": {"u": 3.0}}}
        result = _paired_improvement(baseline, oracle, "test")["u"]
        self.assertAlmostEqual(result["relative_improvement_fraction"], 0.25)
        self.assertEqual(result["per_run_relative_improvement_fraction"], {
            "r1": 0.5, "r2": 0.25})

    def test_windows_require_complete_initial_history_horizon_and_roll(self):
        data = {
            "bounds": np.asarray([[0, 300], [300, 500]]),
            "seq_run": np.asarray([0, 1]),
            "imu_attitude_valid": np.ones(500, dtype=bool),
            "packet_sequence": np.arange(500),
            "dt_s": np.full(500, 0.025),
        }
        rows = _window_rows(data, [{"run": 0, "start": 80}], 200)
        self.assertEqual(rows[0]["start"], 80)
        data["imu_attitude_valid"][120] = False
        self.assertEqual(_window_rows(data, [{"run": 0, "start": 80}], 200), [])
        with self.assertRaises(ValueError):
            _window_rows(data, [{"run": 0, "start": 20}], 200)


if __name__ == "__main__":
    unittest.main()
